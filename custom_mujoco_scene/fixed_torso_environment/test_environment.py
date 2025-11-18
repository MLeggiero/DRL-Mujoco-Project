#!/usr/bin/env python3
"""
Test the G1 environment to verify it's working correctly
"""
import numpy as np
from g1_rl_environment import G1ReachTouchEnv

print("="*70)
print("Testing G1 Reaching Environment")
print("="*70)
print()

# Create environment
env = G1ReachTouchEnv()

print("\nEnvironment created successfully")
print(f"Action dimension: {env.n_actions}")
print()

# Reset environment
print("Resetting environment...")
obs = env.reset()

print(f"Observation keys: {list(obs.keys())}")
print(f"Robot qpos shape: {obs['robot_qpos'].shape}")
print(f"Robot qvel shape: {obs['robot_qvel'].shape}")
print(f"End effector pos: {obs['end_effector_pos']}")
print(f"Target position: {obs['target_position']}")
print(f"Distance to target: {obs['distance_to_target']:.3f}m")
print()

# Test random actions
print("Testing 10 random actions...")
for i in range(10):
    action = np.random.uniform(-1, 1, env.n_actions)
    obs, reward, done, info = env.step(action)

    print(f"Step {i+1}: reward={reward:>7.3f}, distance={obs['distance_to_target']:.3f}m, done={done}")

print()
print("="*70)
print("[SUCCESS] Environment test complete!")
print("="*70)
print()
print("The environment is working correctly.")
print("Next steps:")
print("  1. Install SB3: ./install_sb3.sh")
print("  2. Train: ./train_sb3.sh")
