#!/usr/bin/env python3
"""
Train RL agent to grasp from point cloud observations.

Uses PPO with a custom PointNet-based policy network.
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym
from pointcloud_grasp_env import PointCloudGraspEnv
import argparse
from pathlib import Path


class PointNetExtractor(BaseFeaturesExtractor):
    """
    PointNet-style feature extractor for point cloud observations.

    Architecture:
    1. Per-point MLP to extract local features
    2. Max pooling to get global feature
    3. Concatenate with proprioception
    """

    def __init__(self, observation_space: gym.spaces.Box, point_cloud_size=1024):
        # Calculate feature dimension
        # Point cloud: point_cloud_size * 6 (xyz + rgb)
        # Proprioception: 7 (pos + quat)
        features_dim = 256

        super().__init__(observation_space, features_dim)

        self.point_cloud_size = point_cloud_size

        # Per-point feature extraction (PointNet)
        self.point_net = nn.Sequential(
            nn.Linear(6, 64),  # xyz + rgb -> 64
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Global feature after max pooling: 256
        # Proprioception: 7
        # Combined: 263
        self.feature_fusion = nn.Sequential(
            nn.Linear(256 + 7, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.

        Args:
            observations: Tensor of shape (batch_size, obs_dim)
                         where obs_dim = point_cloud_size * 6 + 7

        Returns:
            features: Tensor of shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        # Split observation into point cloud and proprioception
        point_cloud_flat = observations[:, :-7]  # (batch, point_cloud_size * 6)
        proprioception = observations[:, -7:]     # (batch, 7)

        # Reshape point cloud
        point_cloud = point_cloud_flat.view(batch_size, self.point_cloud_size, 6)

        # Apply PointNet to each point
        point_features = self.point_net(point_cloud)  # (batch, point_cloud_size, 256)

        # Max pooling across points (global feature)
        global_feature, _ = torch.max(point_features, dim=1)  # (batch, 256)

        # Concatenate with proprioception
        combined = torch.cat([global_feature, proprioception], dim=1)  # (batch, 263)

        # Final feature fusion
        features = self.feature_fusion(combined)  # (batch, features_dim)

        return features


class SuccessRateCallback(BaseCallback):
    """
    Callback to track success rate during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_successes = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Get info from all environments
        for info in self.locals.get('infos', []):
            if 'success' in info:
                self.episode_successes.append(float(info['success']))

            # Log episode stats at episode end
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

                # Log to tensorboard
                if len(self.episode_successes) > 0:
                    success_rate = np.mean(self.episode_successes[-100:])
                    self.logger.record('rollout/success_rate', success_rate)

                if len(self.episode_rewards) > 0:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    self.logger.record('rollout/mean_episode_reward', mean_reward)

        return True


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = PointCloudGraspEnv(
            point_cloud_size=1024,
            max_episode_steps=200,
            settle_steps=100
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    """Train PPO agent on point cloud grasping task."""

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Point Cloud Grasping Training")
    print("="*60)
    print(f"Algorithm: PPO")
    print(f"Num environments: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Point cloud size: 1024 points")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Create vectorized environment
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # Policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=PointNetExtractor,
        features_extractor_kwargs=dict(point_cloud_size=1024),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Actor-critic network sizes
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        device='cpu',  # Use CPU for stability
        tensorboard_log=str(output_dir / "tensorboard")
    )

    # Callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.num_envs,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="pointcloud_grasp"
    )
    callbacks.append(checkpoint_callback)

    # Success rate tracking
    success_callback = SuccessRateCallback()
    callbacks.append(success_callback)

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=False  # Disabled to avoid dependency issues
    )

    # Save final model
    final_path = output_dir / "pointcloud_grasp_final.zip"
    model.save(final_path)
    print(f"\nTraining complete! Model saved to {final_path}")

    return model


def evaluate(args):
    """Evaluate trained model."""
    print("="*60)
    print("Evaluating Trained Model")
    print("="*60)

    # Load model
    model = PPO.load(args.model_path)
    print(f"Loaded model from {args.model_path}")

    # Create environment
    env = PointCloudGraspEnv(render_mode='human' if args.visualize else None)

    # Run evaluation episodes
    successes = []
    rewards = []

    for episode in range(args.num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 200:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated

        successes.append(float(info['success']))
        rewards.append(episode_reward)

        print(f"Episode {episode+1}/{args.num_eval_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Success={info['success']}, "
              f"Lift={info['lift_height']:.3f}m")

    # Summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Success rate: {np.mean(successes)*100:.1f}%")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print("="*60)

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for point cloud grasping')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--total-timesteps', type=int, default=1000000,
                             help='Total training timesteps (default: 1M)')
    train_parser.add_argument('--num-envs', type=int, default=4,
                             help='Number of parallel environments (default: 4)')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4,
                             help='Learning rate (default: 3e-4)')
    train_parser.add_argument('--n-steps', type=int, default=2048,
                             help='Steps per environment per update (default: 2048)')
    train_parser.add_argument('--batch-size', type=int, default=64,
                             help='Minibatch size (default: 64)')
    train_parser.add_argument('--n-epochs', type=int, default=10,
                             help='Number of epochs per update (default: 10)')
    train_parser.add_argument('--save-freq', type=int, default=50000,
                             help='Save checkpoint every N steps (default: 50000)')
    train_parser.add_argument('--output-dir', type=str, default='./training_output',
                             help='Output directory (default: ./training_output)')

    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model (.zip)')
    eval_parser.add_argument('--num-eval-episodes', type=int, default=10,
                            help='Number of evaluation episodes (default: 10)')
    eval_parser.add_argument('--visualize', action='store_true',
                            help='Visualize during evaluation')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
