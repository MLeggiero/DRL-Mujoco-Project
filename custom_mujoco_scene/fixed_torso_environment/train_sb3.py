#!/usr/bin/env python3
"""
Train G1 Reaching Task using Stable-Baselines3 PPO
Professional, production-ready RL training with proven algorithms
"""
import os
import argparse
import numpy as np
from datetime import datetime

# Import Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our custom environment
from g1_gym_wrapper import G1ReachingGymEnv


class TrainingProgressCallback:
    """Custom callback to print training progress"""

    def __init__(self, check_freq=1000):
        self.check_freq = check_freq
        self.n_calls = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []

    def __call__(self, locals_dict, globals_dict):
        self.n_calls += 1

        # Get info from the environment
        infos = locals_dict.get('infos', [])

        for info in infos:
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

            if 'distance_to_target' in info:
                self.episode_distances.append(info['distance_to_target'])

        # Print progress every N steps
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_distance = np.mean(self.episode_distances[-100:]) if self.episode_distances else 0

                print(f"Steps: {self.n_calls:>7,} | "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"Avg Length: {avg_length:>5.0f} | "
                      f"Avg Distance: {avg_distance:.3f}m")

        return True


def make_env(scene_path, rank=0, seed=0):
    """Create and wrap the environment"""
    def _init():
        env = G1ReachingGymEnv(scene_path=scene_path)
        env.reset(seed=seed + rank)
        # Wrap with Monitor to track episode statistics
        env = Monitor(env)
        return env
    return _init


def train_g1_reaching(
    scene_path="../unitree_g1/g1_table_box_scene.xml",
    total_timesteps=100_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    save_freq=10_000,
    eval_freq=5_000,
    log_dir="./logs",
    model_dir="./models",
    seed=0,
):
    """
    Train G1 reaching policy using Stable-Baselines3 PPO

    Args:
        scene_path: Path to MuJoCo scene XML
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps to collect per update
        batch_size: Minibatch size for PPO updates
        n_epochs: Number of epochs for PPO updates
        gamma: Discount factor
        save_freq: Save model every N steps
        eval_freq: Evaluate model every N steps
        log_dir: Directory for logs
        model_dir: Directory for saved models
        seed: Random seed
    """

    print("="*70)
    print("STABLE-BASELINES3 PPO TRAINING - G1 REACHING")
    print("="*70)
    print(f"Scene: {scene_path}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"N steps: {n_steps}")
    print(f"N epochs: {n_epochs}")
    print(f"Gamma: {gamma}")
    print("="*70)
    print()

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"g1_ppo_{timestamp}"

    # Create the training environment
    print("Creating training environment...")
    env = DummyVecEnv([make_env(scene_path, rank=0, seed=seed)])

    # Normalize observations and rewards for better training stability
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma
    )

    # Create evaluation environment (without reward normalization)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(scene_path, rank=1, seed=seed + 100)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during evaluation
        clip_obs=10.0,
        training=False
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"{model_dir}/{run_name}",
        name_prefix="g1_ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/{run_name}/best_model",
        log_path=f"{log_dir}/{run_name}",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=f"{log_dir}/{run_name}",
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # 2-layer network with 256 units
        ),
        verbose=1,
        seed=seed,
        device='auto'
    )

    print("\nModel architecture:")
    print(model.policy)
    print()

    # Train the model
    print("="*70)
    print("Starting training...")
    print("="*70)
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True
        )

        print()
        print("="*70)
        print("[SUCCESS] Training complete!")
        print("="*70)

        # Save final model
        final_model_path = f"{model_dir}/{run_name}/final_model"
        model.save(final_model_path)
        env.save(f"{final_model_path}_vecnormalize.pkl")

        print(f"\nFinal model saved to: {final_model_path}")
        print(f"Best model saved to: {model_dir}/{run_name}/best_model")
        print(f"\nTo visualize training progress, run:")
        print(f"  tensorboard --logdir {log_dir}/{run_name}")

        return model, env

    except KeyboardInterrupt:
        print()
        print("="*70)
        print("Training interrupted by user")
        print("="*70)

        # Save interrupted model
        interrupt_path = f"{model_dir}/{run_name}/interrupted_model"
        model.save(interrupt_path)
        env.save(f"{interrupt_path}_vecnormalize.pkl")
        print(f"\nInterrupted model saved to: {interrupt_path}")

        return model, env


def test_trained_model(model_path, scene_path, n_episodes=5):
    """Test a trained model"""
    print("="*70)
    print("TESTING TRAINED MODEL")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("="*70)
    print()

    # Load the model
    model = PPO.load(model_path)

    # Load the normalization stats
    vecnormalize_path = f"{model_path}_vecnormalize.pkl"
    if os.path.exists(vecnormalize_path):
        print("Loading normalization statistics...")

    # Create test environment
    env = DummyVecEnv([make_env(scene_path, rank=0, seed=42)])

    if os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False

    # Test the model
    episode_rewards = []
    episode_distances = []
    episode_successes = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        final_distance = 0

        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1

            if len(info) > 0 and 'distance_to_target' in info[0]:
                final_distance = info[0]['distance_to_target']

        success = final_distance < 0.05
        episode_rewards.append(episode_reward)
        episode_distances.append(final_distance)
        episode_successes.append(success)

        print(f"Episode {episode+1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Distance={final_distance:.3f}m, "
              f"Success={success}")

    print()
    print("="*70)
    print("Test Results:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Average Distance: {np.mean(episode_distances):.3f}m")
    print(f"  Success Rate: {np.mean(episode_successes)*100:.1f}%")
    print("="*70)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train G1 Reaching with Stable-Baselines3 PPO')
    parser.add_argument('--scene', type=str,
                       default="../unitree_g1/g1_table_box_scene.xml",
                       help='Path to scene XML')
    parser.add_argument('--timesteps', type=int, default=100_000,
                       help='Total training timesteps (default: 100k)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Steps per update (default: 2048)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='PPO epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--test', type=str, default=None,
                       help='Test a trained model (provide path)')

    args = parser.parse_args()

    # Check scene file exists
    if not os.path.exists(args.scene):
        print(f"Error: Scene file not found: {args.scene}")
        exit(1)

    # Test mode
    if args.test:
        test_trained_model(args.test, args.scene)
    else:
        # Train mode
        model, env = train_g1_reaching(
            scene_path=args.scene,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            seed=args.seed
        )

        print("\n[SUCCESS] Training pipeline complete!")
