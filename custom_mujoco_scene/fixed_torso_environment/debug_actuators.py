#!/usr/bin/env python3
"""
Debug script to show which actuators are being controlled
"""
import mujoco
import numpy as np
from g1_rl_environment import G1ReachTouchEnv

print("=" * 70)
print("ACTUATOR DEBUG - Which actuators are being controlled?")
print("=" * 70)
print()

env = G1ReachTouchEnv()

print("Total actuators in model:", env.model.nu)
print()

print("=" * 70)
print("CONTROLLABLE ACTUATORS (Right Arm + Torso)")
print("=" * 70)
for i, actuator_id in enumerate(env.controllable_actuators):
    actuator_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    # Get the joint this actuator controls
    joint_id = env.model.actuator_trnid[actuator_id, 0]
    joint_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else "unknown"
    print(f"  Action [{i:2d}] -> Actuator '{actuator_name}' -> Joint '{joint_name}'")

print()
print("=" * 70)
print("LOCKED ACTUATORS (Legs)")
print("=" * 70)
for actuator_id in env.leg_actuators[:10]:  # Show first 10
    actuator_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    joint_id = env.model.actuator_trnid[actuator_id, 0]
    joint_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else "unknown"
    print(f"  Actuator '{actuator_name}' -> Joint '{joint_name}' [LOCKED]")
if len(env.leg_actuators) > 10:
    print(f"  ... and {len(env.leg_actuators) - 10} more leg actuators")

print()
print("=" * 70)
print("TEST: Apply action and see what moves")
print("=" * 70)

# Reset environment
obs = env.reset()

# Get initial joint positions
initial_qpos = env.data.qpos.copy()

# Create a test action: full positive on all controllable actuators
test_action = np.ones(env.n_actions) * 0.5

print(f"Applying test action: {test_action}")
print()

# Take 50 steps with this action
for _ in range(50):
    obs, reward, done, info = env.step(test_action)
    if done:
        break

# Check which joints actually moved
final_qpos = env.data.qpos.copy()
qpos_change = np.abs(final_qpos - initial_qpos)

print("Joint movements after 50 steps:")
print()

# Show controllable joints
print("CONTROLLABLE JOINTS:")
for i in range(env.model.njnt):
    joint_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if joint_name:
        joint_lower = joint_name.lower()

        # Check if this is a right arm or torso joint
        is_controllable = (
            ('right' in joint_lower and any(p in joint_lower for p in ['shoulder', 'elbow', 'wrist', 'arm'])) or
            any(p in joint_lower for p in ['torso', 'waist', 'spine'])
        )

        if is_controllable and env.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            qpos_addr = env.model.jnt_qposadr[i]
            change = qpos_change[qpos_addr]
            status = "✓ MOVED" if change > 0.01 else "✗ STUCK"
            print(f"  {status} {joint_name:40s} Change: {change:8.4f} rad ({np.rad2deg(change):6.2f}°)")

print()
print("=" * 70)
print("DIAGNOSIS")
print("=" * 70)

# Count how many controllable joints actually moved
moved_count = 0
stuck_count = 0

for i in range(env.model.njnt):
    joint_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if joint_name:
        joint_lower = joint_name.lower()
        is_controllable = (
            ('right' in joint_lower and any(p in joint_lower for p in ['shoulder', 'elbow', 'wrist', 'arm'])) or
            any(p in joint_lower for p in ['torso', 'waist', 'spine'])
        )

        if is_controllable and env.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            qpos_addr = env.model.jnt_qposadr[i]
            if qpos_change[qpos_addr] > 0.01:
                moved_count += 1
            else:
                stuck_count += 1

print(f"Controllable joints that MOVED: {moved_count}")
print(f"Controllable joints that are STUCK: {stuck_count}")
print()

if stuck_count > 0:
    print("⚠ WARNING: Some controllable joints are not moving!")
    print("Possible causes:")
    print("  1. Actuator not found/mapped correctly")
    print("  2. Action scaling too low")
    print("  3. Joint limits preventing movement")
    print("  4. Insufficient torque")
else:
    print("✓ All controllable joints are moving as expected")

print()
print("=" * 70)
