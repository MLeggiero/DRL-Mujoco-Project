#!/usr/bin/env python3
"""
Test that leg joints are truly locked during simulation
"""
import numpy as np
from g1_rl_environment import G1ReachTouchEnv

print("="*70)
print("Testing Leg Joint Locking")
print("="*70)
print()

env = G1ReachTouchEnv()
obs = env.reset()

print(f"Number of locked leg joints: {len(env.leg_joint_ids)}")
print(f"Initial leg positions:")
for joint_id, pos in env.initial_leg_qpos.items():
    joint_name = env.model.joint(joint_id).name
    print(f"  {joint_name}: {pos:.4f}")
print()

print("Running 100 random action steps...")
print()

# Track leg positions over time
leg_positions_over_time = {joint_id: [] for joint_id in env.leg_joint_ids}

for step in range(100):
    # Random action
    action = np.random.uniform(-1, 1, env.n_actions)
    obs, reward, done, info = env.step(action)

    # Record leg positions
    for joint_id in env.leg_joint_ids:
        qpos_addr = env.model.jnt_qposadr[joint_id]
        current_pos = env.data.qpos[qpos_addr]
        leg_positions_over_time[joint_id].append(current_pos)

    if done:
        obs = env.reset()

print("="*70)
print("Results: Leg Joint Movement Analysis")
print("="*70)
print()

all_locked = True
for joint_id, positions in leg_positions_over_time.items():
    joint_name = env.model.joint(joint_id).name
    initial = env.initial_leg_qpos[joint_id]
    positions_array = np.array(positions)

    # Check if position ever changed from initial
    max_deviation = np.max(np.abs(positions_array - initial))

    if max_deviation < 1e-6:
        status = "[LOCKED]"
    else:
        status = "[MOVING]"
        all_locked = False

    print(f"{status} {joint_name:30s} Initial: {initial:7.4f}  Max deviation: {max_deviation:.6e}")

print()
print("="*70)
if all_locked:
    print("[SUCCESS] All leg joints are perfectly locked!")
    print("Legs will not move during training or visualization.")
else:
    print("[WARNING] Some leg joints moved during simulation!")
    print("This may cause instability.")
print("="*70)
