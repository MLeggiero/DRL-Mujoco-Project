#!/usr/bin/env python3
"""
Train RL agent for grasping using Grounding DINO vision detection.

This script trains a PPO agent to grasp objects detected with Grounding DINO.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from grounding_dino_grasp_env import GroundingDINOGraspEnv


def make_env(rank, detection_prompt="hammer", use_vision=False, use_geometry_filtering=True):
    """
    Create environment factory for parallel training.
    """
    def _init():
        env = GroundingDINOGraspEnv(
            detection_prompt=detection_prompt,
            use_vision_for_rewards=use_vision,
            use_geometry_filtering=use_geometry_filtering,
            detection_threshold=0.30,
            use_wrist_camera=False,  # State-based for faster training
            max_steps=100
        )
        env = Monitor(env)
        return env
    return _init


def train(
    detection_prompt="hammer",
    use_vision=False,
    use_geometry_filtering=True,
    num_envs=4,
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    output_dir="models/grounding_dino_grasp",
    eval_freq=10_000,
    save_freq=50_000
):
    """
    Train PPO agent for grasping with Grounding DINO detection.

    Args:
        detection_prompt: Text prompt for object detection
        use_vision: If True, use vision detections for rewards (harder)
        use_geometry_filtering: Filter detections by geometry
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        output_dir: Directory to save models
        eval_freq: Evaluate every N steps
        save_freq: Save checkpoint every N steps
    """

    print("="*60)
    print("Training Grounding DINO Grasping Agent")
    print("="*60)
    print(f"Detection prompt: '{detection_prompt}'")
    print(f"Use vision for rewards: {use_vision}")
    print(f"Geometry filtering: {use_geometry_filtering}")
    print(f"Parallel environments: {num_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create vectorized environments
    print(f"\nCreating {num_envs} parallel environments...")

    if num_envs == 1:
        # Single environment
        env = DummyVecEnv([make_env(0, detection_prompt, use_vision, use_geometry_filtering)])
    else:
        # Multiple environments (parallel)
        env = SubprocVecEnv([
            make_env(i, detection_prompt, use_vision, use_geometry_filtering)
            for i in range(num_envs)
        ])

    # Create eval environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(999, detection_prompt, use_vision, use_geometry_filtering)])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // num_envs,  # Per environment
        save_path=str(output_path / "checkpoints"),
        name_prefix="grounding_dino_ppo"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best"),
        log_path=str(output_path / "eval_logs"),
        eval_freq=eval_freq // num_envs,  # Per environment
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Create PPO agent
    print("\nInitializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(output_path / "tensorboard")
    )

    print(f"\nModel architecture:")
    print(model.policy)

    # Train
    print(f"\n{'='*60}")
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

    # Save final model
    final_path = output_path / "final_model.zip"
    model.save(str(final_path))
    print(f"\n✓ Final model saved to: {final_path}")

    # Cleanup
    env.close()
    eval_env.close()

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    return model


def test_trained_model(model_path, num_episodes=10, render=True):
    """
    Test a trained model.

    Args:
        model_path: Path to saved model
        num_episodes: Number of test episodes
        render: Whether to render
    """
    print(f"\nTesting model: {model_path}")

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = GroundingDINOGraspEnv(
        detection_prompt="hammer",
        use_vision_for_rewards=False,
        use_geometry_filtering=True,
        use_wrist_camera=False,
        render_mode="human" if render else None
    )

    # Test episodes
    episode_rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Initial detection: {'✓' if info['detection_success'] else '✗'}")
        if info['detection_success']:
            print(f"  Confidence: {info['detection_confidence']:.1%}")

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if terminated:  # Success
            successes += 1
            print(f"  ✓ SUCCESS in {steps} steps")
        else:
            print(f"  ✗ Failed after {steps} steps")

        print(f"  Reward: {episode_reward:.2f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Test Results ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Success rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"{'='*60}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL agent with Grounding DINO detection")

    # Training args
    parser.add_argument("--prompt", type=str, default="hammer",
                        help="Detection prompt (default: 'hammer')")
    parser.add_argument("--use-vision", action="store_true",
                        help="Use vision detections for rewards (harder, default: False)")
    parser.add_argument("--no-geometry-filter", action="store_true",
                        help="Disable geometry filtering (default: enabled)")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps (default: 1M)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--output-dir", type=str, default="models/grounding_dino_grasp",
                        help="Output directory (default: models/grounding_dino_grasp)")

    # Testing args
    parser.add_argument("--test", type=str, default=None,
                        help="Test a trained model (provide path to .zip)")
    parser.add_argument("--test-episodes", type=int, default=10,
                        help="Number of test episodes (default: 10)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering during test")

    args = parser.parse_args()

    # Test mode
    if args.test:
        test_trained_model(
            args.test,
            num_episodes=args.test_episodes,
            render=not args.no_render
        )
        return

    # Training mode
    train(
        detection_prompt=args.prompt,
        use_vision=args.use_vision,
        use_geometry_filtering=not args.no_geometry_filter,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
