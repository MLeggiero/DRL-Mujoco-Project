#!/usr/bin/env python3
"""
Training script for RGBD-based hammer grasping with Stable-Baselines3.

Demonstrates:
- Vision-only training (CNN-based policy)
- Hybrid vision + proprioception training
- Different observation modes
- Curriculum learning with vision
"""

import argparse
import os
import numpy as np
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback
)
from stable_baselines3.common.monitor import Monitor

from hammer_rgbd_gym_wrapper import HammerRGBDGymWrapper
from hammer_cnn_policies import (
    CNNFeaturesExtractor,
    DualStreamCNNFeaturesExtractor,
    HybridFeaturesExtractor
)


def create_environment(
    obs_mode='rgbd_stacked',
    num_cameras=1,
    image_size=(320, 240),
    use_hand_control=True,
    stack_frames=4
):
    """Create RGBD environment."""
    env = HammerRGBDGymWrapper(
        image_size=image_size,
        num_cameras=num_cameras,
        observation_mode=obs_mode,
        use_hand_control=use_hand_control,
        stack_frames=stack_frames
    )

    return env


def train_vision_only(
    total_timesteps=500000,
    learning_rate=3e-4,
    obs_mode='rgbd_stacked',
    use_hand_control=False,
    num_cameras=1,
    save_dir=None
):
    """Train vision-only policy (CNN-based)."""

    if save_dir is None:
        save_dir = f"./models/rgbd_vision_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: Vision-Only Policy (RGBD)")
    print(f"{'='*60}")
    print(f"Observation mode: {obs_mode}")
    print(f"Cameras: {num_cameras}")
    print(f"Hand control: {use_hand_control}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Save directory: {save_dir}")

    # Create environment
    env = create_environment(
        obs_mode=obs_mode,
        num_cameras=num_cameras,
        use_hand_control=use_hand_control,
        stack_frames=4
    )

    # Wrap with monitoring
    env = Monitor(env)

    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": CNNFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 256]
    }

    # Create model
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix="vision_only"
    )

    progress_callback = ProgressBarCallback()

    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            log_interval=1
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted!")

    # Save final model
    model.save(os.path.join(save_dir, "final_model"))
    print(f"\nModel saved to {save_dir}")

    env.close()


def train_hybrid(
    total_timesteps=500000,
    learning_rate=3e-4,
    use_hand_control=True,
    num_cameras=1,
    save_dir=None
):
    """Train hybrid vision + proprioception policy."""

    if save_dir is None:
        save_dir = f"./models/rgbd_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: Hybrid Vision + Proprioception Policy")
    print(f"{'='*60}")
    print(f"Cameras: {num_cameras}")
    print(f"Hand control: {use_hand_control}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Save directory: {save_dir}")

    # Create environment with raw RGBD (dict observation)
    env = HammerRGBDGymWrapper(
        image_size=(320, 240),
        num_cameras=num_cameras,
        observation_mode='rgbd_raw',  # Dict with rgb, depth, proprioceptive
        use_hand_control=use_hand_control,
        stack_frames=4
    )

    env = Monitor(env)

    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": HybridFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 512,
            "vision_dim": 256,
            "proprio_dim": 128
        },
        "net_arch": [512, 512]
    }

    # Create model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix="hybrid"
    )

    progress_callback = ProgressBarCallback()

    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            log_interval=1
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted!")

    # Save final model
    model.save(os.path.join(save_dir, "final_model"))
    print(f"\nModel saved to {save_dir}")

    env.close()


def evaluate_policy(
    model_path,
    num_episodes=10,
    obs_mode='rgbd_stacked',
    num_cameras=1
):
    """Evaluate trained policy."""

    print(f"\n{'='*60}")
    print(f"Evaluating Policy")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")

    # Create environment
    env = create_environment(
        obs_mode=obs_mode,
        num_cameras=num_cameras
    )

    # Load model
    model = PPO.load(model_path, env=env)

    # Evaluate
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {ep+1}: reward={episode_reward:.4f}, length={episode_length}")

    env.close()

    print(f"\nAverage reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train RGBD hammer grasping policy")

    parser.add_argument('--mode', choices=['vision_only', 'hybrid', 'eval'],
                       default='vision_only', help='Training mode')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--obs-mode', choices=['rgbd_stacked', 'rgb', 'depth'],
                       default='rgbd_stacked', help='Observation mode')
    parser.add_argument('--cameras', type=int, choices=[1, 2], default=1,
                       help='Number of cameras')
    parser.add_argument('--hand-control', action='store_true',
                       help='Enable hand control')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Save directory for models')
    parser.add_argument('--eval-model', type=str, default=None,
                       help='Model path for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')

    args = parser.parse_args()

    if args.mode == 'vision_only':
        train_vision_only(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            obs_mode=args.obs_mode,
            use_hand_control=args.hand_control,
            num_cameras=args.cameras,
            save_dir=args.save_dir
        )

    elif args.mode == 'hybrid':
        train_hybrid(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            use_hand_control=args.hand_control,
            num_cameras=args.cameras,
            save_dir=args.save_dir
        )

    elif args.mode == 'eval':
        if args.eval_model is None:
            print("Error: --eval-model required for evaluation mode")
            return 1

        evaluate_policy(
            model_path=args.eval_model,
            num_episodes=args.eval_episodes,
            obs_mode=args.obs_mode,
            num_cameras=args.cameras
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
