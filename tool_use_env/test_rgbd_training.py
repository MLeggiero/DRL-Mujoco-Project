#!/usr/bin/env python3
"""
Quick test script to validate RGBD environment and start training.
"""

import os
import sys

# Set OpenGL backend before importing MuJoCo
os.environ['MUJOCO_GL'] = 'egl'  # Try EGL first (fastest)

import numpy as np
from hammer_rgbd_gym_wrapper import HammerRGBDGymWrapper

def test_environment():
    """Test if the RGBD environment works."""

    print("="*60)
    print("Testing RGBD Environment")
    print("="*60)

    try:
        # Create environment with minimal settings
        print("\n1. Creating environment...")
        env = HammerRGBDGymWrapper(
            observation_mode='rgbd_raw',  # Dict observation for hybrid
            num_cameras=1,
            use_hand_control=False,  # Simpler action space
            stack_frames=4
        )

        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Test reset
        print("\n2. Testing reset...")
        obs, info = env.reset()

        if isinstance(obs, dict):
            print(f"✓ Reset successful - Dict observation:")
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    {key}: {value}")
        else:
            print(f"✓ Reset successful - Array observation: shape={obs.shape}")

        # Test a few steps
        print("\n3. Testing environment steps...")
        total_reward = 0
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"  Step {i+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")

            if terminated or truncated:
                print(f"  Episode ended, resetting...")
                obs, info = env.reset()

        print(f"\n✓ Environment test passed!")
        print(f"  Total reward: {total_reward:.3f}")

        env.close()

        return True

    except Exception as e:
        print(f"\n✗ Environment test failed!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_setup():
    """Test if training dependencies are available."""

    print("\n" + "="*60)
    print("Testing Training Dependencies")
    print("="*60)

    try:
        print("\n1. Checking Stable-Baselines3...")
        import stable_baselines3
        print(f"✓ Stable-Baselines3 version: {stable_baselines3.__version__}")

        print("\n2. Checking PyTorch...")
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        print("\n3. Checking custom policies...")
        from hammer_cnn_policies import CNNFeaturesExtractor, HybridFeaturesExtractor
        print(f"✓ Custom policies available")

        print("\n4. Checking training script...")
        train_script = os.path.join(os.path.dirname(__file__), 'train_rgbd_sb3.py')
        if os.path.exists(train_script):
            print(f"✓ Training script found: {train_script}")
        else:
            print(f"✗ Training script not found!")
            return False

        return True

    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("\nInstall missing packages with:")
        print("  pip install stable-baselines3 torch")
        return False


def print_next_steps():
    """Print recommended next steps."""

    print("\n" + "="*60)
    print("READY TO TRAIN!")
    print("="*60)

    print("\nRecommended training commands:\n")

    print("1. HYBRID MODEL (Vision + Proprioception) - Fastest learning:")
    print("   python3 train_rgbd_sb3.py \\")
    print("       --mode hybrid \\")
    print("       --obs-mode rgbd_raw \\")
    print("       --cameras 1 \\")
    print("       --timesteps 1000000 \\")
    print("       --save-dir ./models/hybrid_v1\n")

    print("2. VISION-ONLY MODEL - Pure visual learning:")
    print("   python3 train_rgbd_sb3.py \\")
    print("       --mode vision_only \\")
    print("       --obs-mode rgbd_stacked \\")
    print("       --cameras 1 \\")
    print("       --timesteps 2000000 \\")
    print("       --save-dir ./models/vision_v1\n")

    print("3. EVALUATE A TRAINED MODEL:")
    print("   python3 train_rgbd_sb3.py \\")
    print("       --mode eval \\")
    print("       --eval-model ./models/hybrid_v1/final_model.zip \\")
    print("       --eval-episodes 100\n")

    print("Expected training time:")
    print("  - 1M steps: 6-12 hours (depending on hardware)")
    print("  - 2M steps: 12-24 hours")
    print("\nExpected success rate:")
    print("  - Hybrid model: 70-85%")
    print("  - Vision-only: 60-70%")
    print("="*60)


def main():
    """Run all tests."""

    print("\n" + "="*60)
    print("RGBD Training Environment Validation")
    print("="*60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version.split()[0]}")

    # Test environment
    env_ok = test_environment()

    if not env_ok:
        print("\n⚠ Environment test failed. Cannot proceed with training.")
        print("\nTroubleshooting:")
        print("1. If you see OpenGL errors, try:")
        print("   export MUJOCO_GL=osmesa")
        print("   (or add to your ~/.bashrc)")
        print("\n2. If you're on WSL, you may need:")
        print("   sudo apt-get install libosmesa6-dev")
        print("\n3. Check that the XML scene file exists")
        return 1

    # Test training setup
    train_ok = test_training_setup()

    if not train_ok:
        print("\n⚠ Training dependencies missing. Install them first.")
        return 1

    # Print next steps
    print_next_steps()

    return 0


if __name__ == "__main__":
    sys.exit(main())
