#!/usr/bin/env python3
"""
Improved SB3 Training Configuration for G1 Reaching
Based on analysis of first training run
"""
import os
import argparse
from train_sb3 import train_g1_reaching, test_trained_model

# OPTIMIZED hyperparameters for FAST CONVERGENCE + SMOOTH MOTION
# Key improvements: Better reward shaping, velocity penalties, learning rate schedule
IMPROVED_CONFIG = {
    # Training length - optimized for faster convergence
    "total_timesteps": 1_000_000,  # 1M steps should now converge faster with improved rewards

    # Learning parameters - ADAPTIVE schedule for faster early learning
    "learning_rate": 5e-4,  # Higher initial LR (was 3e-4) for faster convergence
    # Note: Will decay automatically with PPO's built-in schedule if using linear schedule

    # PPO parameters - optimized for faster convergence
    "n_steps": 2048,  # Collect diverse experience
    "batch_size": 256,  # INCREASED from 128 - larger batches for more stable updates
    "n_epochs": 10,  # Standard PPO range

    # Discount and GAE - tuned for reaching task
    "gamma": 0.99,  # Standard discount
    "gae_lambda": 0.95,  # Helps with credit assignment

    # Monitoring - more frequent for faster convergence tracking
    "save_freq": 10_000,  # Save every 10k steps (was 20k)
    "eval_freq": 5_000,   # Evaluate every 5k steps (was 10k) - catch success early

    # Device
    "device": "cuda",  # Use GPU by default

    # Parallel environments for faster training
    "n_envs": 4,  # Use 4 parallel environments (safe for most systems)

    # Action smoothing parameters - OPTIMIZED for smooth motion
    "action_smoothing": 0.2,  # INCREASED smoothing (was 0.3) - smoother motion
    "smoothness_weight": 3.0,  # INCREASED (was 2.0) - stronger smoothness penalty
    "action_scale": 0.35,      # INCREASED (was 0.3) - slightly faster movements for convergence
    "sim_substeps": 10,        # Good balance between stability and speed
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved G1 training with smooth control')
    parser.add_argument('--scene', type=str,
                       default="../unitree_g1/g1_table_box_scene.xml",
                       help='Path to scene XML')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--n-envs', type=int, default=None,
                       help='Number of parallel environments (default: 8)')
    parser.add_argument('--action-smoothing', type=float, default=None,
                       help='Action smoothing EMA coefficient (0-1, lower=smoother)')
    parser.add_argument('--smoothness-weight', type=float, default=None,
                       help='Weight for smoothness penalty in reward')
    parser.add_argument('--action-scale', type=float, default=None,
                       help='Action scaling factor (lower=smoother but slower)')
    parser.add_argument('--sim-substeps', type=int, default=None,
                       help='Number of physics substeps per RL step')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable real-time plotting (useful for headless systems)')
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
        print("OPTIMIZED TRAINING - FAST CONVERGENCE + SMOOTH MOTION")
        print("="*70)
        print("\nConfiguration for SMOOTH GENERALIZED REACHING:")
        print("  - Object position RANDOMIZED each episode!")
        print("    * X: 0.35-0.65m (forward)")
        print("    * Y: -0.2 to +0.2m (left/right)")
        print("  - Legs perfectly locked")
        print()
        # Override config with command-line arguments BEFORE printing
        config = IMPROVED_CONFIG.copy()
        config['device'] = args.device
        config['enable_plot'] = not args.no_plot

        # Override smoothness parameters if provided
        if args.n_envs is not None:
            config['n_envs'] = args.n_envs
        if args.action_smoothing is not None:
            config['action_smoothing'] = args.action_smoothing
        if args.smoothness_weight is not None:
            config['smoothness_weight'] = args.smoothness_weight
        if args.action_scale is not None:
            config['action_scale'] = args.action_scale
        if args.sim_substeps is not None:
            config['sim_substeps'] = args.sim_substeps

        print("CONVERGENCE OPTIMIZATIONS:")
        print(f"  - Higher learning rate: {config['learning_rate']} (faster early learning)")
        print(f"  - Larger batch size: {config['batch_size']} (more stable updates)")
        print(f"  - Enhanced reward shaping: exponential + linear distance rewards")
        print(f"  - Stronger progress rewards: 50x gradient (always active)")
        print(f"  - Earlier proximity bonuses: starting at 20cm (was 15cm)")
        print(f"  - MASSIVE success bonus: +2000 (dominates all penalties)")
        print(f"  - More frequent eval: every {config['eval_freq']} steps")
        print()
        print("SMOOTHNESS IMPROVEMENTS:")
        print(f"  - Action filtering: EMA alpha={config['action_smoothing']} (STRONGER)")
        print(f"  - Smoothness penalty: weight={config['smoothness_weight']} (INCREASED)")
        print(f"  - Velocity penalties: -0.01 * joint_vel² (NEW!)")
        print(f"  - Action scaling: {config['action_scale']}")
        print(f"  - Physics substeps: {config['sim_substeps']}")
        print(f"  - Lower entropy coefficient: 0.0005 (stable exploration)")
        print(f"  - Larger network: 512x512 (was 256x256)")
        print(f"  - target_kl: 0.02 (prevents divergence)")
        print()
        print("PARALLEL TRAINING:")
        print(f"  - Parallel environments: {config['n_envs']} (speeds up data collection)")
        print()
        print("VISUALIZATION:")
        print(f"  - Real-time plotting: {'Enabled' if config['enable_plot'] else 'Disabled'}")
        print()
        print("TRAINING PARAMETERS:")
        print(f"  - Total timesteps: {config['total_timesteps']:,}")
        print(f"  - Episode length: 400 steps")
        print(f"  - Rollout length: {config['n_steps']} steps")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - Epochs: {config['n_epochs']}")
        print("="*70)
        print()
        print("ENHANCED REWARD FUNCTION:")
        print("  Distance:")
        print("    - Exponential + linear: -5*d + 10*exp(-10*d) [steeper decay]")
        print("  Progress:")
        print("    - Always active: 50x gradient")
        print("    - Penalty for moving away: 3x stronger")
        print("  Proximity bonuses (cumulative):")
        print("    - 20cm: +10 | 15cm: +20 | 10cm: +50 | 8cm: +100 | 6cm: +200")
        print("    - <5cm: +2000 ⭐ GOAL ACHIEVEMENT - dominates all other rewards!")
        print("  Smoothness:")
        print("    - Velocity penalty: -0.01 * Σ(joint_vel²)")
        print("    - Action change penalty: -3.0 * Σ(Δaction²)")
        print("    - Action magnitude: -0.005 * Σ(action²)")
        print("="*70)
        print()

        model, env = train_g1_reaching(
            scene_path=args.scene,
            **config
        )

        print("\n[SUCCESS] Improved training complete!")
