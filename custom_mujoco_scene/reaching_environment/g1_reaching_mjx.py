"""
G1 Reaching Environment using MuJoCo Playground (MJX) with GPU Acceleration
This version uses GPU-accelerated physics simulation and training

Installation:
  pip install playground
  pip install --upgrade "jax[cuda12]"  # For NVIDIA CUDA 12
  # OR
  pip install --upgrade "jax[cuda11]"  # For NVIDIA CUDA 11
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, grad
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import Dict, Tuple
import pickle

# MuJoCo Playground imports
try:
    from mujoco import mjx
    import mujoco
    print(f"✓ MJX available (GPU physics)")
    print(f"✓ JAX backend: {jax.default_backend()}")
    print(f"✓ JAX devices: {jax.devices()}")
except ImportError as e:
    print(f"✗ Error importing MJX: {e}")
    print("Install with: pip install playground")
    raise

# Check for GPU
if jax.default_backend() != 'gpu':
    print("⚠ WARNING: No GPU detected! Running on CPU.")
    print("Install CUDA-enabled JAX: pip install --upgrade 'jax[cuda12]'")


class G1ReachingEnvMJX:
    """GPU-accelerated G1 reaching environment using MJX"""
    
    def __init__(self, xml_path='g1_table_box_scene.xml', 
                 num_envs=128,  # Batch size for parallel simulation
                 episode_length=500):
        """
        Args:
            xml_path: Path to MuJoCo XML file
            num_envs: Number of parallel environments (GPU batch size)
            episode_length: Maximum steps per episode
        """
        self.xml_path = xml_path
        self.num_envs = num_envs
        self.episode_length = episode_length
        
        # Load MuJoCo model
        print(f"Loading model from {xml_path}...")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Create MJX model (GPU version)
        print(f"Creating MJX model for {num_envs} parallel environments...")
        self.model = mjx.put_model(self.mj_model)
        
        # Cache body IDs
        self._setup_ids()
        
        # JIT compile step function
        self.step_fn = jit(self._step_fn)
        self.reset_fn = jit(self._reset_fn)
        
        print(f"✓ Environment ready with {num_envs} parallel envs on {jax.devices()[0]}")
    
    def _setup_ids(self):
        """Cache body/site IDs"""
        try:
            self.right_hand_id = self.mj_model.site('right_hand_site').id
            self.use_site = True
        except:
            self.right_hand_id = self.mj_model.body('right_wrist_yaw_link').id
            self.use_site = False
        
        self.red_box_id = self.mj_model.body('red_box').id
        self.pelvis_id = self.mj_model.body('pelvis').id
        
        print(f"Using {'site' if self.use_site else 'body'} for hand tracking")
    
    def _reset_fn(self, rng):
        """Reset environment (JIT compiled, runs on GPU)"""
        # Create initial data state
        data = mjx.make_data(self.model)
        
        # Add small random noise to initial state
        qpos_noise = random.uniform(rng, shape=(self.model.nq,), minval=-0.01, maxval=0.01)
        qvel_noise = random.uniform(rng, shape=(self.model.nv,), minval=-0.01, maxval=0.01)
        
        data = data.replace(
            qpos=data.qpos + qpos_noise,
            qvel=data.qvel + qvel_noise
        )
        
        # Forward kinematics
        data = mjx.forward(self.model, data)
        
        return data
    
    def reset(self, rng):
        """Reset all parallel environments"""
        # Generate separate RNG keys for each environment
        rngs = random.split(rng, self.num_envs)
        
        # Vectorized reset across all environments
        data_batch = vmap(self.reset_fn)(rngs)
        
        return data_batch
    
    def _step_fn(self, data, action):
        """Single step (JIT compiled, runs on GPU)"""
        # Apply action
        data = data.replace(ctrl=action)
        
        # Step physics (5 substeps for smooth simulation)
        for _ in range(5):
            data = mjx.step(self.model, data)
        
        return data
    
    def step(self, data_batch, actions):
        """Step all parallel environments"""
        # Vectorized step across all environments
        data_batch = vmap(self.step_fn)(data_batch, actions)
        
        # Compute observations and rewards
        obs_batch = self._get_obs(data_batch)
        rewards = self._compute_rewards(data_batch)
        dones = self._check_done(data_batch)
        
        return data_batch, obs_batch, rewards, dones
    
    def _get_obs(self, data_batch):
        """Get observations for all environments (vectorized)"""
        def single_obs(data):
            if self.use_site:
                hand_pos = data.site_xpos[self.right_hand_id]
            else:
                hand_pos = data.xpos[self.right_hand_id]
            
            target_pos = data.xpos[self.red_box_id]
            
            obs = jnp.concatenate([
                data.qpos,
                data.qvel,
                hand_pos,
                target_pos,
                hand_pos - target_pos
            ])
            return obs
        
        return vmap(single_obs)(data_batch)
    
    def _compute_rewards(self, data_batch):
        """Compute rewards for all environments (vectorized)"""
        def single_reward(data):
            if self.use_site:
                hand_pos = data.site_xpos[self.right_hand_id]
            else:
                hand_pos = data.xpos[self.right_hand_id]
            
            target_pos = data.xpos[self.red_box_id]
            distance = jnp.linalg.norm(hand_pos - target_pos)
            
            # Main reward: negative distance
            reward = -distance
            
            # Bonus for getting very close
            reward += jnp.where(distance < 0.05, 10.0, 0.0)
            
            # Small action penalty
            reward -= 0.001 * jnp.sum(data.ctrl ** 2)
            
            return reward
        
        return vmap(single_reward)(data_batch)
    
    def _check_done(self, data_batch):
        """Check if episodes are done (vectorized)"""
        def single_done(data):
            # Check if robot fell
            pelvis_height = data.xpos[self.pelvis_id][2]
            fallen = pelvis_height < 0.4
            
            # Check if reached target
            if self.use_site:
                hand_pos = data.site_xpos[self.right_hand_id]
            else:
                hand_pos = data.xpos[self.right_hand_id]
            target_pos = data.xpos[self.red_box_id]
            distance = jnp.linalg.norm(hand_pos - target_pos)
            reached = distance < 0.03
            
            return jnp.logical_or(fallen, reached)
        
        return vmap(single_done)(data_batch)
    
    @property
    def observation_dim(self):
        """Get observation dimension"""
        return self.mj_model.nq + self.mj_model.nv + 9  # qpos + qvel + hand + target + relative
    
    @property
    def action_dim(self):
        """Get action dimension"""
        return self.mj_model.nu


class PolicyNetwork(nn.Module):
    """Policy network using Flax (JAX neural network library)"""
    action_dim: int
    hidden_dims: Tuple[int] = (256, 256)
    
    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.tanh(x)
        
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)  # Bound actions to [-1, 1]
        
        return x


class ValueNetwork(nn.Module):
    """Value network using Flax"""
    hidden_dims: Tuple[int] = (256, 256)
    
    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.tanh(x)
        
        x = nn.Dense(1)(x)
        
        return x.squeeze()


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation (GPU accelerated)"""
    def scan_fn(carry, transition):
        next_value, last_advantage = carry
        reward, value, done = transition
        
        delta = reward + gamma * next_value * (1 - done) - value
        advantage = delta + gamma * lam * (1 - done) * last_advantage
        
        return (value, advantage), advantage
    
    # Reverse trajectories
    rewards_reversed = jnp.flip(rewards, axis=0)
    values_reversed = jnp.flip(values, axis=0)
    dones = jnp.zeros_like(rewards)  # Simplified for now
    
    _, advantages = jax.lax.scan(
        scan_fn,
        (0.0, 0.0),
        (rewards_reversed, values_reversed, dones)
    )
    
    advantages = jnp.flip(advantages, axis=0)
    returns = advantages + values
    
    return advantages, returns


def train_ppo_mjx(
    xml_path='g1_table_box_scene.xml',
    num_envs=128,
    num_epochs=100,
    steps_per_epoch=500,
    lr_policy=3e-4,
    lr_value=1e-3,
    gamma=0.99,
    lam=0.95,
    clip_epsilon=0.2,
    num_minibatches=4,
    update_epochs=4,
    save_path='ppo_mjx_policy.pkl'
):
    """
    Train PPO with GPU-accelerated MJX simulation
    
    Args:
        num_envs: Number of parallel environments (higher = better GPU utilization)
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per environment per epoch
    """
    
    print("="*60)
    print("PPO Training with MuJoco Playground (GPU Accelerated)")
    print("="*60)
    print(f"Configuration:")
    print(f"  Parallel Environments: {num_envs}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total timesteps per epoch: {num_envs * steps_per_epoch}")
    print(f"  Policy LR: {lr_policy}")
    print(f"  Value LR: {lr_value}")
    print(f"  Devices: {jax.devices()}")
    print("="*60)
    
    # Initialize environment
    env = G1ReachingEnvMJX(xml_path, num_envs=num_envs, episode_length=steps_per_epoch)
    
    # Initialize networks
    rng = random.PRNGKey(0)
    rng, policy_rng, value_rng = random.split(rng, 3)
    
    dummy_obs = jnp.zeros((env.observation_dim,))
    
    policy_net = PolicyNetwork(action_dim=env.action_dim)
    policy_params = policy_net.init(policy_rng, dummy_obs)
    
    value_net = ValueNetwork()
    value_params = value_net.init(value_rng, dummy_obs)
    
    # Create optimizers
    policy_optimizer = optax.adam(lr_policy)
    policy_opt_state = policy_optimizer.init(policy_params)
    
    value_optimizer = optax.adam(lr_value)
    value_opt_state = value_optimizer.init(value_params)
    
    # Training loop
    best_reward = -float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # Reset environments
        rng, reset_rng = random.split(rng)
        data_batch = env.reset(reset_rng)
        
        # Collect trajectories
        all_obs = []
        all_actions = []
        all_rewards = []
        all_values = []
        all_dones = []
        
        print(f"Collecting {steps_per_epoch} steps from {num_envs} parallel environments...")
        
        for step in range(steps_per_epoch):
            # Get current observations
            obs_batch = env._get_obs(data_batch)
            
            # Get actions from policy
            actions = policy_net.apply(policy_params, obs_batch)
            
            # Get values
            values = value_net.apply(value_params, obs_batch)
            
            # Step environments (all parallel on GPU)
            data_batch, next_obs, rewards, dones = env.step(data_batch, actions)
            
            # Store data
            all_obs.append(obs_batch)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_values.append(values)
            all_dones.append(dones)
            
            if (step + 1) % 100 == 0:
                avg_reward = jnp.mean(rewards)
                print(f"  Step {step+1}/{steps_per_epoch}: Avg reward = {avg_reward:.2f}")
        
        # Stack trajectories
        all_obs = jnp.stack(all_obs)  # Shape: (steps, num_envs, obs_dim)
        all_actions = jnp.stack(all_actions)
        all_rewards = jnp.stack(all_rewards)
        all_values = jnp.stack(all_values)
        all_dones = jnp.stack(all_dones)
        
        # Compute advantages (vectorized over environments)
        advantages_list = []
        returns_list = []
        
        for env_idx in range(num_envs):
            adv, ret = compute_gae(
                all_rewards[:, env_idx],
                all_values[:, env_idx],
                gamma, lam
            )
            advantages_list.append(adv)
            returns_list.append(ret)
        
        advantages = jnp.stack(advantages_list, axis=1)  # (steps, num_envs)
        returns = jnp.stack(returns_list, axis=1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten batches for training
        obs_flat = all_obs.reshape(-1, env.observation_dim)
        actions_flat = all_actions.reshape(-1, env.action_dim)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        
        dataset_size = obs_flat.shape[0]
        batch_size = dataset_size // num_minibatches
        
        print(f"Training on {dataset_size} timesteps...")
        
        # PPO update
        for update_epoch in range(update_epochs):
            # Shuffle data
            rng, shuffle_rng = random.split(rng)
            perm = random.permutation(shuffle_rng, dataset_size)
            
            for i in range(num_minibatches):
                batch_indices = perm[i * batch_size:(i + 1) * batch_size]
                
                obs_batch = obs_flat[batch_indices]
                actions_batch = actions_flat[batch_indices]
                advantages_batch = advantages_flat[batch_indices]
                returns_batch = returns_flat[batch_indices]
                
                # Policy loss and update
                def policy_loss_fn(params):
                    pred_actions = policy_net.apply(params, obs_batch)
                    action_diff = (pred_actions - actions_batch) ** 2
                    loss = jnp.mean(jnp.sum(action_diff, axis=-1) * advantages_batch)
                    return loss
                
                policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
                updates, policy_opt_state = policy_optimizer.update(policy_grads, policy_opt_state)
                policy_params = optax.apply_updates(policy_params, updates)
                
                # Value loss and update
                def value_loss_fn(params):
                    pred_values = value_net.apply(params, obs_batch)
                    loss = jnp.mean((pred_values - returns_batch) ** 2)
                    return loss
                
                value_loss, value_grads = jax.value_and_grad(value_loss_fn)(value_params)
                updates, value_opt_state = value_optimizer.update(value_grads, value_opt_state)
                value_params = optax.apply_updates(value_params, updates)
        
        # Log results
        epoch_rewards = all_rewards.mean(axis=0)  # Average over time
        avg_reward = float(epoch_rewards.mean())
        max_reward = float(epoch_rewards.max())
        
        if max_reward > best_reward:
            best_reward = max_reward
            print(f"Epoch {epoch + 1}: NEW BEST! Avg={avg_reward:.2f}, Max={max_reward:.2f}")
            
            # Save best policy
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'policy_params': policy_params,
                    'value_params': value_params,
                    'observation_dim': env.observation_dim,
                    'action_dim': env.action_dim
                }, f)
            print(f"✓ Saved to {save_path}")
        else:
            print(f"Epoch {epoch + 1}: Avg={avg_reward:.2f}, Max={max_reward:.2f}, Best={best_reward:.2f}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Policy saved to: {save_path}")
    print("="*60)
    
    return policy_params, value_params


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train G1 reaching with GPU-accelerated MJX')
    parser.add_argument('--xml_path', type=str, default='g1_table_box_scene.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--num_envs', type=int, default=128,
                        help='Number of parallel environments (GPU batch size)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=500,
                        help='Steps per environment per epoch')
    parser.add_argument('--lr_policy', type=float, default=3e-4,
                        help='Policy learning rate')
    parser.add_argument('--lr_value', type=float, default=1e-3,
                        help='Value learning rate')
    parser.add_argument('--save', type=str, default='ppo_mjx_policy.pkl',
                        help='Path to save trained policy')
    
    args = parser.parse_args()
    
    # Train
    policy_params, value_params = train_ppo_mjx(
        xml_path=args.xml_path,
        num_envs=args.num_envs,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        save_path=args.save
    )