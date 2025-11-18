#!/usr/bin/env python3
"""
Auto-detect hardware and run appropriate training
This script chooses the best training method based on available hardware
"""
import os
import sys
import subprocess

def check_gpu():
    """Check if GPU is available and get VRAM"""
    try:
        import jax
        devices = jax.devices()

        if any('gpu' in str(d).lower() for d in devices):
            print("[OK] GPU detected via JAX")
            # Try to estimate VRAM (rough heuristic)
            return True, "unknown"
        else:
            print("[INFO] No GPU detected (CPU only)")
            return False, None
    except:
        print("[INFO] JAX not available or no GPU")
        return False, None

def main():
    print("="*70)
    print("G1 Reaching Task - Auto Training")
    print("="*70)
    print()

    # Detect hardware
    print("Detecting hardware...")
    has_gpu, vram = check_gpu()
    print()

    # Choose training method
    if has_gpu:
        print("Recommendation: Use Brax with GPU acceleration")
        print()
        print("Choose training intensity:")
        print("  1. Light    (128 envs, 2M steps, ~1-2 hours)")
        print("  2. Medium   (512 envs, 5M steps, ~1-2 hours)")
        print("  3. Heavy    (2048 envs, 10M steps, ~30-60 min)")
        print("  4. Custom   (specify your own parameters)")
        print("  5. PPO      (single environment, CPU fallback)")
        print()

        choice = input("Enter choice [1-5] (default: 1): ").strip() or "1"

        if choice == "1":
            cmd = [
                "python", "train_with_brax.py",
                "--timesteps", "2000000",
                "--episode_length", "300",
                "--num_envs", "128",
                "--lr", "3e-4"
            ]
        elif choice == "2":
            cmd = [
                "python", "train_with_brax.py",
                "--timesteps", "5000000",
                "--episode_length", "400",
                "--num_envs", "512",
                "--lr", "3e-4"
            ]
        elif choice == "3":
            cmd = [
                "python", "train_with_brax.py",
                "--timesteps", "10000000",
                "--episode_length", "500",
                "--num_envs", "2048",
                "--lr", "3e-4"
            ]
        elif choice == "4":
            timesteps = input("Timesteps (default: 2000000): ").strip() or "2000000"
            num_envs = input("Num environments (default: 128): ").strip() or "128"
            episode_length = input("Episode length (default: 300): ").strip() or "300"

            cmd = [
                "python", "train_with_brax.py",
                "--timesteps", timesteps,
                "--episode_length", episode_length,
                "--num_envs", num_envs,
                "--lr", "3e-4"
            ]
        else:  # choice == "5" or invalid
            cmd = [
                "python", "ppo_training.py",
                "--epochs", "100",
                "--episodes_per_epoch", "10",
                "--max_steps", "300"
            ]
    else:
        print("Recommendation: Use lightweight PPO (CPU optimized)")
        print()
        print("This will run standard PPO with CPU-friendly settings:")
        print("  - 100 epochs")
        print("  - 10 episodes per epoch")
        print("  - 300 steps per episode")
        print("  - Expected time: 2-4 hours")
        print()

        proceed = input("Proceed with PPO training? [Y/n]: ").strip().lower()
        if proceed and proceed != 'y' and proceed != 'yes':
            print("Training cancelled.")
            return

        cmd = [
            "python", "ppo_training.py",
            "--epochs", "100",
            "--episodes_per_epoch", "10",
            "--max_steps", "300",
            "--save", "ppo_g1_policy_cpu.pkl"
        ]

    print()
    print("="*70)
    print("Starting training with command:")
    print(" ".join(cmd))
    print("="*70)
    print()

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print()
        print("="*70)
        print("[SUCCESS] Training completed successfully!")
        print("="*70)
    except subprocess.CalledProcessError as e:
        print()
        print("="*70)
        print(f"[ERROR] Training failed with error code {e.returncode}")
        print("="*70)
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("="*70)
        print("Training interrupted by user")
        print("="*70)
        sys.exit(0)

if __name__ == "__main__":
    main()
