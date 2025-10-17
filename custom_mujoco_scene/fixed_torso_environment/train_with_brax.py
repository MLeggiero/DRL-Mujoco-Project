"""
Brax PPO Training for G1 Reaching Task
Uses proven Brax implementation with GPU acceleration
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from jax import random
import mujoco
from mujoco import mjx
import numpy as np
from typing import Any, Dict, Tuple
import functools
import time

print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

try:
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from brax import envs
    from brax.envs.base import State, Env, PipelineEnv
    print("✅ Brax imported successfully")
except ImportError as e:
    print(f"❌ Error importing Brax: {e}")
    print("\nInstalling required packages...")
    print("Run: pip install brax==0.10.5 jax jaxlib")
    exit(1)


# ============================================================================
# G1 Reaching Environment for Brax
# ============================================================================

class G1ReachingBraxEnv(PipelineEnv):
    """
    G1 Reaching environment compatible with Brax training
    """
    
    def __init__(
        self,
        scene_path: str = "../unitree_g1/g1_table_box_scene.xml",
        backend: str = 'mjx',
        **kwargs
    ):
        """Initialize G1 Reaching environment"""
        
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")
        
        print(f"\n{'='*70}")
        print("Initializing G1 Reaching Environment for Brax")
        print(f"{'='*70}")
        print(f"Scene: {scene_path}")
        print(f"Backend: {backend}")
        
        # Load MuJoCo model
        mj_model = mujoco.MjModel.from_xml_path(scene_path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        
        # Find important sites
        try:
            self.hand_site_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_palm"
            )
            print(f"✅ Found hand site: right_palm (id={self.hand_site_id})")
        except:
            print("⚠️  Warning: 'right_palm' site not found, searching for alternatives...")
            # Try to find any hand-related site
            for i in range(mj_model.nsite):
                site_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
                if site_name and ('hand' in site_name.lower() or 'palm' in site_name.lower()):
                    self.hand_site_id = i
                    print(f"✅ Using site: {site_name} (id={i})")
                    break
        
        # Find target body
        try:
            self.target_body_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, "red_box"
            )
            print(f"✅ Found target: red_box (id={self.target_body_id})")
        except:
            print("❌ Error: 'red_box' body not found")
            raise
        
        # Find controllable actuators (right arm + torso)
        self.controllable_indices = []
        for i in range(mj_model.nu):
            act_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name and any(kw in act_name for kw in [
                'right_shoulder', 'right_elbow', 'right_wrist',
                'waist_yaw', 'waist_roll', 'waist_pitch'
            ]):
                self.controllable_indices.append(i)
        
        print(f"✅ Controllable actuators: {len(self.controllable_indices)}/{mj_model.nu}")
        print(f"{'='*70}\n")
        
        sys = mjx.put_model(mj_model)
        
        # Initialize parent with n_frames=1 for direct control
        super().__init__(sys, backend=backend, n_frames=1)
    
    def reset(self, rng: jax.Array) -> State:
        """Reset environment"""
        # Create initial state with small noise
        rng, rng1, rng2 = jax.random.split(rng, 3)
        
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=-0.01, maxval=0.01
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=-0.01, maxval=0.01
        )
        
        pipeline_state = self.pipeline_init(qpos, qvel)
        
        obs = self._get_obs(pipeline_state)
        reward, done = jnp.zeros(2)
        metrics = {
            'distance': jnp.float32(0),
            'success': jnp.float32(0),
        }
        
        return State(pipeline_state, obs, reward, done, metrics)
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step environment"""
        # Create full control vector
        ctrl = jnp.zeros(self.sys.nu)
        ctrl = ctrl.at[jnp.array(self.controllable_indices)].set(action)
        
        # Step physics
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        
        # Get observation
        obs = self._get_obs(pipeline_state)
        
        # Calculate reward
        hand_pos = pipeline_state.site_xpos[self.hand_site_id]
        target_pos = pipeline_state.xpos[self.target_body_id]
        distance = jnp.linalg.norm(hand_pos - target_pos)
        
        # Dense reward: negative distance + bonuses
        reward = -distance
        reward = jnp.where(distance < 0.1, reward + 1.0, reward)
        reward = jnp.where(distance < 0.05, reward + 5.0, reward)
        
        # Small penalty for large actions
        action_penalty = 0.001 * jnp.sum(action ** 2)
        reward = reward - action_penalty
        
        # Check success
        success = distance < 0.05
        done = success
        
        metrics = {
            'distance': distance,
            'success': success.astype(jnp.float32),
        }
        
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics
        )
    
    def _get_obs(self, pipeline_state) -> jax.Array:
        """Get observation"""
        # Joint positions and velocities (skip free joint)
        qpos = pipeline_state.qpos[7:]  # Skip free joint (7 DOF)
        qvel = pipeline_state.qvel[6:]  # Skip free joint velocities (6 DOF)
        
        # Hand and target positions
        hand_pos = pipeline_state.site_xpos[self.hand_site_id]
        target_pos = pipeline_state.xpos[self.target_body_id]
        
        # Relative vector
        hand_to_target = target_pos - hand_pos
        
        # Concatenate observation
        obs = jnp.concatenate([
            qpos,
            qvel,
            hand_pos,
            target_pos,
            hand_to_target,
        ])
        
        return obs
    
    @property
    def observation_size(self) -> int:
        """Observation dimension"""
        nq = self.sys.nq - 7  # Exclude free joint
        nv = self.sys.nv - 6  # Exclude free joint velocities
        return nq + nv + 3 + 3 + 3  # + hand_pos + target_pos + hand_to_target
    
    @property
    def action_size(self) -> int:
        """Action dimension"""
        return len(self.controllable_indices)


# ============================================================================
# Training Function
# ============================================================================

def train_g1_reaching(
    scene_path: str = "../unitree_g1/g1_table_box_scene.xml",
    num_timesteps: int = 10_000_000,
    num_evals: int = 10,
    episode_length: int = 500,
    num_envs: int = 2048,
    learning_rate: float = 3e-4,
    seed: int = 0,
):
    """
    Train G1 reaching policy using Brax PPO
    
    Args:
        scene_path: Path to MuJoCo scene XML
        num_timesteps: Total training timesteps
        num_evals: Number of evaluation checkpoints
        episode_length: Max steps per episode
        num_envs: Number of parallel environments (increase for GPU)
        learning_rate: Learning rate
        seed: Random seed
    """
    
    print("\n" + "="*70)
    print("BRAX PPO TRAINING - G1 REACHING")
    print("="*70)
    print(f"Total timesteps: {num_timesteps:,}")
    print(f"Episode length: {episode_length}")
    print(f"Parallel environments: {num_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {jax.devices()[0]}")
    print(f"Backend: {jax.default_backend()}")
    print("="*70)
    
    # Create environment
    env = G1ReachingBraxEnv(scene_path=scene_path)
    
    # Wrap with training wrappers
    env = envs.wrappers.training.EpisodeWrapper(env, episode_length, action_repeat=1)
    env = envs.wrappers.training.AutoResetWrapper(env)
    
    # Training callback
    times = [time.time()]
    
    def progress(num_steps, metrics):
        times.append(time.time())
        dt = times[-1] - times[-2] if len(times) > 1 else 0
        steps_per_sec = (num_timesteps // num_evals) / dt if dt > 0 else 0
        
        # Get metrics
        reward = metrics.get('eval/episode_reward', 0)
        distance = metrics.get('eval/episode_distance', 0)
        success = metrics.get('eval/episode_success', 0)
        
        print(
            f"Step {num_steps:>8,} | "
            f"Reward: {reward:>7.2f} | "
            f"Distance: {distance:.4f}m | "
            f"Success: {success:>5.1%} | "
            f"SPS: {steps_per_sec:>6,.0f}"
        )
    
    print("\n🚀 Starting Brax PPO training...")
    print("This uses GPU/TPU acceleration via JAX\n")
    
    # Train
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        reward_scaling=1.0,
        episode_length=episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=learning_rate,
        entropy_cost=1e-2,
        num_envs=num_envs,
        batch_size=1024,
        seed=seed,
        progress_fn=progress,
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    # Save policy
    os.makedirs("./brax_policies", exist_ok=True)
    policy_path = "./brax_policies/g1_reaching_policy.pkl"
    
    import pickle
    with open(policy_path, 'wb') as f:
        pickle.dump({
            'params': params,
            'make_inference_fn': make_inference_fn,
        }, f)
    
    print(f"\n💾 Policy saved to: {policy_path}")
    
    return make_inference_fn, params


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train G1 Reaching with Brax PPO')
    parser.add_argument('--scene', type=str,
                       default="../unitree_g1/g1_table_box_scene.xml",
                       help='Path to scene XML')
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                       help='Total training timesteps (default: 10M)')
    parser.add_argument('--episode_length', type=int, default=500,
                       help='Max steps per episode')
    parser.add_argument('--num_envs', type=int, default=2048,
                       help='Number of parallel environments')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Check scene file exists
    if not os.path.exists(args.scene):
        print(f"❌ Error: Scene file not found: {args.scene}")
        exit(1)
    
    # Train
    make_inference_fn, params = train_g1_reaching(
        scene_path=args.scene,
        num_timesteps=args.timesteps,
        episode_length=args.episode_length,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        seed=args.seed,
    )
    
    print("\n✨ Done! Your policy is ready to use.")