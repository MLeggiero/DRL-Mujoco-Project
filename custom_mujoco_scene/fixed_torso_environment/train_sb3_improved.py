#!/usr/bin/env python3
"""
Improved SB3 Training Configuration for G1 Reaching
Based on analysis of first training run
"""
import os
import argparse
from train_sb3 import train_g1_reaching, test_trained_model

# Improved hyperparameters for fixed environment
# With objects at 0.5m, moderate action scaling (0.5), and fixed reward function
IMPROVED_CONFIG = {
    # Training length - increased due to reward function fix
    "total_timesteps": 1_000_000,  # 100k steps (reward gaming fix requires more exploration)

    # Learning parameters
    "learning_rate": 4.7e-4,  # Higher for faster learning (was 3e-4)

    # PPO parameters
    "n_steps": 2048,  # Longer rollouts for stable gradient estimates
    "batch_size": 128,  # Larger batches for stability
    "n_epochs": 25,  # More gradient steps per update

    # Discount and GAE
    "gamma": 0.99,  # Standard discount

    # Monitoring
    "save_freq": 20_000,  # Save every 20k steps
    "eval_freq": 10_000,  # Evaluate every 10k steps
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved G1 training')
    parser.add_argument('--scene', type=str,
                       default="../unitree_g1/g1_table_box_scene.xml",
                       help='Path to scene XML')
    parser.add_argument('--test', type=str, default=None,
                       help='Test a trained model')

    args = parser.parse_args()

    if not os.path.exists(args.scene):
        print(f"Error: Scene file not found: {args.scene}")
        exit(1)

    if args.test:
        test_trained_model(args.test, args.scene)
    else:
        print("="*70)
        print("IMPROVED TRAINING CONFIGURATION")
        print("="*70)
        print("\nConfiguration for GENERALIZED REACHING:")
        print("  - Object position RANDOMIZED each episode!")
        print("    * X: 0.35-0.65m (forward)")
        print("    * Y: -0.2 to +0.2m (left/right)")
        print("  - Legs perfectly locked")
        print("  - Strong action scaling (0.4) for full arm extension")
        print("  - Close-range focused rewards")
        print("  - 500k timesteps (more needed for generalization)")
        print("  - 400 step episodes")
        print("  - Higher learning rate (5e-4)")
        print("  - 2048 step rollouts, 128 batch size")
        print("="*70)
        print()
        print("REWARD FUNCTION - EXTREME CLOSE-RANGE FOCUS:")
        print("  - Distance: -10 * d (linear, provides basic gradient)")
        print("  - Progress reward: 20x but ONLY when distance < 20cm")
        print("  - Proximity bonuses (HEAVILY weighted to very close):")
        print("    * 15cm: +5 (small)")
        print("    * 10cm: +20 (getting bigger)")
        print("    * 8cm:  +50 (large)")
        print("    * 6cm:  +100 (huge)")
        print("    * 5cm:  +300 (MASSIVE SUCCESS)")
        print("  - Total possible: +475 for success!")
        print("  - Almost all rewards at distances < 10cm")
        print("="*70)
        print()

        model, env = train_g1_reaching(
            scene_path=args.scene,
            **IMPROVED_CONFIG
        )

        print("\n[SUCCESS] Improved training complete!")
