#!/usr/bin/env python3
"""
Training script for vision-guided grasping with multiple strategies.

Supports:
1. Physics-only baseline (fast training)
2. Pure vision (harder, more realistic)
3. Curriculum learning (physics → vision)
4. Hybrid approach (best of both)
"""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch

from vision_guided_grasp_env import VisionGuidedGraspEnv


def make_env(rank, config):
    """Create environment factory."""
    def _init():
        env = VisionGuidedGraspEnv(
            target_object=config['target_object'],
            use_vision_detection=config['use_vision'],
            reward_mode=config['reward_mode'],
            reward_shaping=config['reward_shaping'],
            observation_mode=config['obs_mode'],
            detection_threshold=config['detection_threshold'],
            add_noise=config['add_noise'],
            max_steps=config['max_steps'],
            vision_update_freq=config.get('vision_update_freq', 1)
        )
        env = Monitor(env)
        return env
    return _init


def train_baseline(args):
    """
    Strategy 1: Physics-only baseline.

    Fast training with perfect information.
    Good for validating RL setup and reward function.
    """
    print("\n" + "="*60)
    print("Strategy 1: Physics-Only Baseline")
    print("="*60)

    config = {
        'target_object': args.target,
        'use_vision': False,  # Physics only
        'reward_mode': 'physics',
        'reward_shaping': 'dense',
        'obs_mode': 'state',
        'detection_threshold': 0.30,
        'add_noise': False,
        'max_steps': 100,
        'vision_update_freq': args.vision_freq
    }

    output_dir = Path(args.output_dir) / "baseline_physics"
    return _run_training(config, args, output_dir, "Baseline (Physics)")


def train_pure_vision(args):
    """
    Strategy 2: Pure vision.

    Uses only vision detections (no ground truth).
    Harder to train but more realistic.
    """
    print("\n" + "="*60)
    print("Strategy 2: Pure Vision")
    print("="*60)

    config = {
        'target_object': args.target,
        'use_vision': True,
        'reward_mode': 'vision',  # Vision only
        'reward_shaping': 'dense',
        'obs_mode': 'state',
        'detection_threshold': 0.30,
        'add_noise': False,
        'max_steps': 100,
        'vision_update_freq': args.vision_freq
    }

    output_dir = Path(args.output_dir) / "pure_vision"
    return _run_training(config, args, output_dir, "Pure Vision")


def train_curriculum(args):
    """
    Strategy 3: Curriculum learning.

    Phase 1: Train with physics (500K steps)
    Phase 2: Fine-tune with vision (500K steps)
    """
    print("\n" + "="*60)
    print("Strategy 3: Curriculum Learning")
    print("="*60)

    output_dir = Path(args.output_dir) / "curriculum"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Physics training
    print("\n--- Phase 1: Training with Physics ---")
    config_phase1 = {
        'target_object': args.target,
        'use_vision': False,
        'reward_mode': 'physics',
        'reward_shaping': 'dense',
        'obs_mode': 'state',
        'detection_threshold': 0.30,
        'add_noise': False,
        'max_steps': 100,
        'vision_update_freq': args.vision_freq
    }

    model_phase1 = _run_training(
        config_phase1,
        args,
        output_dir / "phase1",
        "Curriculum Phase 1",
        timesteps=args.timesteps // 2
    )

    # Phase 2: Vision fine-tuning
    print("\n--- Phase 2: Fine-tuning with Vision ---")
    config_phase2 = {
        'target_object': args.target,
        'use_vision': True,
        'reward_mode': 'vision',
        'reward_shaping': 'dense',
        'obs_mode': 'state',
        'detection_threshold': 0.30,
        'add_noise': True,  # Add noise for robustness
        'max_steps': 100,
        'vision_update_freq': args.vision_freq
    }

    # Create new environment with vision
    if args.num_envs == 1:
        env = DummyVecEnv([make_env(0, config_phase2)])
    else:
        env = SubprocVecEnv([
            make_env(i, config_phase2)
            for i in range(args.num_envs)
        ])

    # Load phase 1 model and continue training
    print("Loading Phase 1 model...")
    model = PPO.load(
        str(output_dir / "phase1" / "final_model.zip"),
        env=env
    )

    print(f"Continuing training with vision for {args.timesteps // 2} steps...")

    # Setup callbacks
    eval_env = DummyVecEnv([make_env(999, config_phase2)])

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // args.num_envs,
        save_path=str(output_dir / "phase2" / "checkpoints"),
        name_prefix="vision_finetune"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "phase2" / "best"),
        log_path=str(output_dir / "phase2" / "eval_logs"),
        eval_freq=10000 // args.num_envs,
        deterministic=True,
        n_eval_episodes=5
    )

    model.learn(
        total_timesteps=args.timesteps // 2,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save final model
    model.save(str(output_dir / "phase2" / "final_model.zip"))
    print(f"✓ Phase 2 complete! Model saved to {output_dir / 'phase2'}")

    env.close()
    eval_env.close()

    return model


def train_hybrid(args):
    """
    Strategy 4: Hybrid approach.

    Uses both vision and physics:
    - Vision detection for target finding
    - Physics for reward calculation (70% vision, 30% physics)
    - Best balance of realism and training speed
    """
    print("\n" + "="*60)
    print("Strategy 4: Hybrid Approach")
    print("="*60)

    config = {
        'target_object': args.target,
        'use_vision': True,
        'reward_mode': 'hybrid',  # Mix vision + physics
        'reward_shaping': 'dense',
        'obs_mode': 'state',
        'detection_threshold': 0.30,
        'add_noise': False,
        'max_steps': 100,
        'vision_update_freq': args.vision_freq
    }

    output_dir = Path(args.output_dir) / "hybrid"
    return _run_training(config, args, output_dir, "Hybrid")


def _run_training(config, args, output_dir, strategy_name, timesteps=None):
    """Internal training function."""
    if timesteps is None:
        timesteps = args.timesteps

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create environments
    print(f"\nCreating {args.num_envs} parallel environments...")
    if args.num_envs == 1:
        env = DummyVecEnv([make_env(0, config)])
    else:
        env = SubprocVecEnv([
            make_env(i, config)
            for i in range(args.num_envs)
        ])

    # Create eval environment
    eval_env = DummyVecEnv([make_env(999, config)])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // args.num_envs,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=f"{strategy_name.lower().replace(' ', '_')}_ppo"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=10000 // args.num_envs,
        deterministic=True,
        n_eval_episodes=5
    )

    # Create model
    print(f"\nInitializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nTraining {strategy_name} for {timesteps:,} steps...")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

    # Save final model
    model.save(str(output_dir / "final_model.zip"))
    print(f"\n✓ {strategy_name} training complete!")
    print(f"Model saved to: {output_dir / 'final_model.zip'}")

    env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train vision-guided grasping with multiple strategies"
    )

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=["baseline", "vision", "curriculum", "hybrid", "all"],
        help="Training strategy (default: hybrid)"
    )

    # Basic settings
    parser.add_argument("--target", type=str, default="hammer",
                       help="Target object (default: hammer)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total timesteps (default: 1M)")
    parser.add_argument("--num-envs", type=int, default=4,
                       help="Parallel environments (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--output-dir", type=str, default="models/vision_grasp",
                       help="Output directory (default: models/vision_grasp)")
    parser.add_argument("--vision-freq", type=int, default=10,
                       help="Vision update frequency in episodes (default: 10, use 1 for every episode)")

    args = parser.parse_args()

    print("="*60)
    print("Vision-Guided Grasping Training")
    print("="*60)
    print(f"Strategy: {args.strategy}")
    print(f"Target: {args.target}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Run selected strategy
    if args.strategy == "baseline":
        train_baseline(args)
    elif args.strategy == "vision":
        train_pure_vision(args)
    elif args.strategy == "curriculum":
        train_curriculum(args)
    elif args.strategy == "hybrid":
        train_hybrid(args)
    elif args.strategy == "all":
        print("\nTraining all strategies...")
        train_baseline(args)
        train_pure_vision(args)
        train_hybrid(args)
        train_curriculum(args)

    print("\n" + "="*60)
    print("✓ All training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
