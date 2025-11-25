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
    "total_timesteps": 12_000_000,  # 1M steps should now converge faster with improved rewards

    # Learning parameters - Lower for more stable learning
    "learning_rate": 3e-4,  # Slightly lower for more stable learning
    # Note: Will decay automatically with PPO's built-in schedule if using linear schedule

    # PPO parameters - optimized for breaking local minima
    "n_steps": 4096,  # INCREASED for better advantage estimation
    "batch_size": 512,  # INCREASED for more stable gradients
    "n_epochs": 10,  # Standard PPO range
    "ent_coef": 0.01,  # INCREASED exploration (from 0.0035)
    "clip_range": 0.1,  # Tighter clipping for fine control
    "target_kl": 0.01,  # More conservative policy updates

    # Discount and GAE - tuned for reaching task
    "gamma": 0.98,  # Standard discount
    "gae_lambda": 0.95,  # Helps with credit assignment

    # Monitoring - more frequent for faster convergence tracking
    "save_freq": 10_000,  # Save every 10k steps (was 20k)
    "eval_freq": 5_000,   # Evaluate every 5k steps (was 10k) - catch success early

    # Device
    "device": "cpu",  # Use CPU by default

    # Parallel environments for faster training
    "n_envs": 4,  # Use 4 parallel environments (safe for most systems)

    # Action smoothing parameters - BALANCED for reaching
    "action_smoothing": 0.2,  # INCREASED smoothing (was 0.3) - smoother motion
    "smoothness_weight": 1.5,  # REDUCED to allow bolder actions (was 3.0)
    "action_scale": 0.5,       # INCREASED to 0.5 for precise control (was 0.28)
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
        print("    * X: 0.2-0.5m (forward) - 10CM CLOSER for initial training!")
        print("    * Y: ±10cm (left/right) - EXACTLY ±10CM VARIATION")
        print("    * Z: Fixed at table height (no vertical variation)")
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

        print("BREAKING BOTH 0.3M AND 0.1M LOCAL MINIMA - ENHANCED REWARDS:")
        print()
        print("1. REWARD FUNCTION IMPROVEMENTS (V2 - Enhanced for 0.1m plateau):")
        print(f"  - Exponential distance reward: exp(-15*d) - STRONGER gradient (was -10)")
        print(f"  - MASSIVE success bonus: +5000 at <5cm (was +1000 - 5x increase!)")
        print(f"  - TIERED Comfort zone penalties:")
        print(f"    • >0.15m: -100 (breaks 0.3m plateau)")
        print(f"    • 0.08-0.15m: -80 (NEW! breaks 0.1m plateau)")
        print(f"    • 0.06-0.08m: -40 (final push to goal)")
        print(f"  - Proximity bonuses: <0.15m:+50, <0.10m:+100, <0.08m:+200, <0.06m:+500")
        print(f"  - Velocity reward: +10*v_toward_target (encourages active movement)")
        print(f"  - Dense approach bonus: +200*(Δd) every step (was +100 - DOUBLED)")
        print()
        print("2. ACTION SCALE INCREASE:")
        print(f"  - Action scaling: {config['action_scale']} (DOUBLED from 0.25 for precise control)")
        print()
        print("3. CURRICULUM LEARNING (10cm closer for initial success!):")
        print(f"  - Episodes 0-1000: Easy targets (0.2-0.4m, ±10cm)")
        print(f"  - Episodes 1000-3000: Medium targets (0.3-0.5m, ±10cm)")
        print(f"  - Episodes 3000+: Full range (0.3-0.5m, ±10cm)")
        print(f"  - Red box is 10cm closer to enable successful initial training!")
        print()
        print("4. PPO PARAMETER TUNING:")
        print(f"  - Learning rate: {config['learning_rate']} (stable learning)")
        print(f"  - Batch size: {config['batch_size']} (INCREASED for stable gradients)")
        print(f"  - Rollout steps: {config['n_steps']} (INCREASED for better advantage estimation)")
        print(f"  - Entropy coefficient: {config['ent_coef']} (INCREASED exploration)")
        print(f"  - Clip range: {config['clip_range']} (tighter for fine control)")
        print(f"  - Target KL: {config['target_kl']} (conservative policy updates)")
        print()
        print("5. SUCCESS-BASED RESET:")
        print(f"  - After success: robot position varies by ±2cm for generalization")
        print()
        print("6. SMOOTHNESS SETTINGS:")
        print(f"  - Action filtering: EMA alpha={config['action_smoothing']}")
        print(f"  - Smoothness penalty: weight={config['smoothness_weight']}")
        print(f"  - Physics substeps: {config['sim_substeps']}")
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

        model, env = train_g1_reaching(
            scene_path=args.scene,
            **config
        )

        print("\n[SUCCESS] Improved training complete!")
