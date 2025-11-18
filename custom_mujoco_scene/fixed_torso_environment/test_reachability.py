#!/usr/bin/env python3
"""
Test if the target is reachable with fixed base configuration
Uses random actions to see maximum reach
"""
import numpy as np
from g1_rl_environment import G1ReachTouchEnv

print("="*70)
print("Testing Target Reachability with Fixed Base")
print("="*70)
print()

env = G1ReachTouchEnv()
obs = env.reset()

print(f"Initial configuration:")
print(f"  Hand position: {obs['end_effector_pos']}")
print(f"  Target position: {obs['target_position']}")
print(f"  Initial distance: {obs['distance_to_target']:.3f}m")
print()

# Test with random actions to see if we can get closer
min_distance = obs['distance_to_target']
best_hand_pos = obs['end_effector_pos'].copy()
best_action = None

print("Testing random arm movements (1000 samples)...")
print()

for i in range(1000):
    # Reset to initial pose
    obs = env.reset()

    # Try a random sustained action for 50 steps
    action = np.random.uniform(-1, 1, env.n_actions)

    for _ in range(50):
        obs, reward, done, info = env.step(action)

        distance = obs['distance_to_target']
        if distance < min_distance:
            min_distance = distance
            best_hand_pos = obs['end_effector_pos'].copy()
            best_action = action.copy()
            print(f"New best distance: {min_distance:.3f}m (trial {i+1})")

print()
print("="*70)
print("Results")
print("="*70)
print(f"Initial distance: 0.467m")
print(f"Minimum distance achieved: {min_distance:.3f}m")
print(f"Improvement: {(0.467 - min_distance):.3f}m")
print()

if min_distance < 0.05:
    print("[SUCCESS] Target IS reachable! (got within 5cm)")
    print("The robot can physically reach the target with fixed base.")
elif min_distance < 0.15:
    print("[PARTIAL SUCCESS] Got close (within 15cm)")
    print("Target is reachable with better control.")
elif min_distance < 0.30:
    print("[PROGRESS] Got moderately close (within 30cm)")
    print("Target appears reachable with proper training.")
else:
    print("[ISSUE] Could not get very close")
    print("Target might be at edge of workspace or unreachable.")

print()
print("Best hand position:", best_hand_pos)
print("Target position:", obs['target_position'])
print()

if min_distance > 0.20:
    print("Recommendation: Target might be too far for fixed base.")
    print("Options:")
    print("  1. Move target closer in XML")
    print("  2. Adjust initial robot position")
    print("  3. Allow base to move (not recommended for learning)")
else:
    print("Recommendation: Keep fixed base, improve training.")
    print("  - Apply quick fixes: ./apply_quick_fixes.sh")
    print("  - Retrain: python train_sb3_improved.py")
