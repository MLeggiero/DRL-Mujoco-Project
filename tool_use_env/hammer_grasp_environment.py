#!/usr/bin/env python3
"""
Unitree G1 with Hands Reinforcement Learning Environment for Hammer Grasping Tasks.

This module implements a MuJoCo-based RL environment for training a Unitree G1
humanoid robot with hand manipulation capabilities to grasp and manipulate a hammer.
The robot has a fixed pelvis and controlled arms/hands for grasping the hammer on a table.
"""

import mujoco
import numpy as np
import time
from typing import Dict, Tuple, Optional
import os

class HammerGraspEnv:
    """MuJoCo environment for Unitree G1 hammer grasping tasks.

    Implements a robot grasping task where the G1 humanoid uses its hands
    to grasp a hammer on a table and perform manipulation tasks.
    """

    def __init__(self,
                 scene_path="hammer_grasp_scene.xml",
                 action_smoothing=0.3,
                 smoothness_weight=1.0,
                 action_scale=0.5,
                 sim_substeps=10,
                 use_hand_control=True):
        """Initialize the hammer grasping environment.

        Args:
            scene_path: Path to MuJoCo XML scene file
            action_smoothing: EMA coefficient for action filtering (0=max smoothing, 1=no filtering)
            smoothness_weight: Weight for action smoothness penalty in reward function
            action_scale: Scaling factor for actuator commands
            sim_substeps: Number of physics simulation steps per environment step
            use_hand_control: Whether to include hand/finger actuators in action space
        """
        self.scene_path = scene_path
        print(f"Using scene file: {self.scene_path}")

        if not os.path.exists(self.scene_path):
            raise FileNotFoundError(f"Scene file not found: {self.scene_path}")

        self.model = None
        self.data = None
        self.viewer = None

        # Episode parameters
        self.max_episode_steps = 800  # Extended time for grasping tasks
        self.current_step = 0

        # Robot configuration
        self.initial_robot_pos = [0.0, 0.0, 0.793]  # Fixed pelvis height

        # Grasping target
        self.hammer_body_name = "hammer"
        self.target_sites = {
            'hammer_head': 'hammer_head',
            'hammer_grasp': 'hammer_grasp_point'
        }

        # Action smoothing configuration
        self.action_smoothing = action_smoothing
        self.smoothness_weight = smoothness_weight
        self.action_scale = action_scale
        self.sim_substeps = sim_substeps
        self.filtered_action = None
        self.use_hand_control = use_hand_control

        # Reward tracking
        self.grasp_success_distance = 0.05  # Distance threshold for successful grasp
        self.last_distance = None
        self.last_action = None
        self.last_hammer_pos = None
        self.last_end_effector_pos = None
        self.contact_with_hammer = False
        self.grasp_contact_frames = 0

        # Episode tracking
        self.episode_count = 0
        self.last_episode_success = False

        print(f"Action smoothing: alpha={action_smoothing}, weight={smoothness_weight}")
        print(f"Action scale: {action_scale}, Substeps: {sim_substeps}")
        print(f"Hand control enabled: {use_hand_control}")

        # Observation space indices (will be set after model loads)
        self.right_arm_qpos_indices = []
        self.left_arm_qpos_indices = []
        self.hand_qpos_indices = []
        self.right_arm_qvel_indices = []
        self.left_arm_qvel_indices = []
        self.hand_qvel_indices = []

        self._load_model()
        self._setup_actuators()

    def _load_model(self):
        """Load the MuJoCo model from the scene file."""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.scene_path)
            self.data = mujoco.MjData(self.model)
            print(f"Loaded model: {self.model.nbody} bodies, {self.model.njnt} joints, {self.model.nu} actuators")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _setup_actuators(self):
        """Identify and map actuators for arm/hand control.

        Separates controllable actuators (arms and hands) from leg actuators,
        which are locked to maintain a stable base during grasping tasks.
        """
        self.controllable_actuators = []
        self.leg_actuators = []
        self.leg_joint_ids = []
        self.initial_leg_qpos = {}

        # Identify arm and hand actuators, lock legs
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                actuator_lower = actuator_name.lower()

                # Lock all leg actuators
                if any(leg_part in actuator_lower for leg_part in ['hip', 'knee', 'ankle', 'leg']):
                    self.leg_actuators.append(i)
                    joint_id = self.model.actuator_trnid[i, 0]
                    if joint_id >= 0 and joint_id not in self.leg_joint_ids:
                        self.leg_joint_ids.append(joint_id)
                    continue

                # Control both arm actuators (left and right)
                if any(arm_part in actuator_lower for arm_part in ['shoulder', 'elbow', 'wrist', 'arm']):
                    self.controllable_actuators.append(i)
                    continue

                # Control hand/finger actuators if enabled
                if self.use_hand_control and any(hand_part in actuator_lower for hand_part in ['hand', 'thumb', 'index', 'middle']):
                    self.controllable_actuators.append(i)
                    continue

                # Control torso/waist actuators
                if any(torso_part in actuator_lower for torso_part in ['torso', 'waist', 'spine']):
                    self.controllable_actuators.append(i)

        print(f"Controllable actuators (arms/hands/torso): {len(self.controllable_actuators)} actuators")
        print(f"Locked leg actuators: {len(self.leg_actuators)} actuators")
        print(f"Locked leg joints: {len(self.leg_joint_ids)} joints")

        self.n_actions = len(self.controllable_actuators)
        if self.n_actions == 0:
            print("WARNING: No arm/hand/torso actuators found - using first 10 actuators")
            self.n_actions = min(self.model.nu, 10)
            self.controllable_actuators = list(range(self.n_actions))

        # Map actuator names for debugging
        print(f"\nControllable actuators (first 10):")
        for i, act_idx in enumerate(self.controllable_actuators[:10]):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_idx)
            print(f"  {i}: {name}")

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state.

        Returns:
            Initial observation
        """
        # Reset data
        mujoco.mj_resetData(self.model, self.data)

        # Set robot to initial position (fixed base)
        robot_pos = self.initial_robot_pos
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id >= 0:
            self.data.xpos[pelvis_id] = robot_pos

        # Set initial leg joint positions for stability
        for joint_id in self.leg_joint_ids:
            if joint_id >= 0 and joint_id < self.model.nq:
                # Set legs to neutral position
                self.data.qpos[joint_id] = 0.0

        # Reset hammer position with slight variations
        hammer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.hammer_body_name)
        if hammer_id >= 0:
            # Randomize hammer position slightly on table
            base_pos = np.array([0.5, 0.0, 0.75])
            noise = np.random.uniform(-0.05, 0.05, 3)
            noise[2] = 0  # Don't randomize height
            hammer_pos = base_pos + noise
            self.data.xpos[hammer_id][:] = hammer_pos

        # Perform forward kinematics
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.filtered_action = np.zeros(self.n_actions)
        self.last_action = np.zeros(self.n_actions)
        self.contact_with_hammer = False
        self.grasp_contact_frames = 0

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get the current observation from the environment.

        Returns:
            Observation array containing robot state and task information
        """
        # Get right hand end-effector position
        right_palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        if right_palm_id < 0:
            # Fallback: get right wrist position
            right_wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
            if right_wrist_id >= 0:
                right_hand_pos = self.data.xpos[right_wrist_id]
            else:
                right_hand_pos = np.zeros(3)
        else:
            right_hand_pos = self.data.site_xpos[right_palm_id]

        # Get left hand end-effector position
        left_palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
        if left_palm_id < 0:
            left_wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
            if left_wrist_id >= 0:
                left_hand_pos = self.data.xpos[left_wrist_id]
            else:
                left_hand_pos = np.zeros(3)
        else:
            left_hand_pos = self.data.site_xpos[left_palm_id]

        # Get hammer position
        hammer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.hammer_body_name)
        if hammer_id >= 0:
            hammer_pos = self.data.xpos[hammer_id]
            hammer_vel = self.data.cvel[hammer_id][:3]
        else:
            hammer_pos = np.zeros(3)
            hammer_vel = np.zeros(3)

        # Get arm joint positions and velocities
        arm_qpos = []
        arm_qvel = []
        for i, qpos_idx in enumerate(range(self.model.nq)):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) if i < self.model.njnt else None
            if joint_name and ('shoulder' in joint_name or 'elbow' in joint_name or 'wrist' in joint_name):
                arm_qpos.append(self.data.qpos[qpos_idx])
                if qpos_idx < len(self.data.qvel):
                    arm_qvel.append(self.data.qvel[qpos_idx])

        if not arm_qpos:
            arm_qpos = np.zeros(14)
            arm_qvel = np.zeros(14)
        else:
            arm_qpos = np.array(arm_qpos[:14])
            arm_qvel = np.array(arm_qvel[:14])

        # Build observation
        obs = np.concatenate([
            right_hand_pos,           # 3: right hand position
            left_hand_pos,            # 3: left hand position
            hammer_pos,               # 3: hammer position
            hammer_vel,               # 3: hammer velocity
            arm_qpos,                 # 14: arm joint positions
            arm_qvel,                 # 14: arm joint velocities
        ])

        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step of the environment.

        Args:
            action: Control commands for controllable actuators

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Apply action smoothing with exponential moving average
        if self.filtered_action is None:
            self.filtered_action = action.copy()
        else:
            self.filtered_action = (self.action_smoothing * action +
                                   (1 - self.action_smoothing) * self.filtered_action)

        # Scale action
        scaled_action = self.filtered_action * self.action_scale

        # Lock leg actuators to initial positions
        for leg_act_idx in self.leg_actuators:
            joint_id = self.model.actuator_trnid[leg_act_idx, 0]
            if joint_id >= 0:
                target_pos = 0.0  # Keep legs at neutral position
                self.data.ctrl[leg_act_idx] = target_pos

        # Apply arm/hand control actions
        for i, act_idx in enumerate(self.controllable_actuators):
            if i < len(scaled_action):
                self.data.ctrl[act_idx] = scaled_action[i]

        # Step simulation multiple times for stability
        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)

        # Check contact with hammer
        hammer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.hammer_body_name)
        hand_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"right_hand_{part}_link")
            for part in ['index_0', 'middle_0', 'thumb_0']
        ]
        hand_ids = [bid for bid in hand_ids if bid >= 0]

        self.contact_with_hammer = False
        if hammer_id >= 0 and hand_ids:
            for contact in self.data.contact:
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]

                if (geom1_body == hammer_id and geom2_body in hand_ids) or \
                   (geom2_body == hammer_id and geom1_body in hand_ids):
                    self.contact_with_hammer = True
                    break

        # Calculate reward
        reward = self._compute_reward(action)

        # Check termination conditions
        done = self.current_step >= self.max_episode_steps
        self.current_step += 1

        # Prepare info dict
        info = {
            'step': self.current_step,
            'contact_with_hammer': self.contact_with_hammer,
            'episode_success': False,
        }

        obs = self._get_observation()

        return obs, reward, done, info

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute the reward for the current step.

        Args:
            action: The action taken in this step

        Returns:
            Reward value
        """
        reward = 0.0

        # Get hand and hammer positions
        right_palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        if right_palm_id >= 0:
            right_hand_pos = self.data.site_xpos[right_palm_id]
        else:
            right_wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
            right_hand_pos = self.data.xpos[right_wrist_id] if right_wrist_id >= 0 else np.zeros(3)

        hammer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.hammer_body_name)
        if hammer_id >= 0:
            hammer_pos = self.data.xpos[hammer_id]
        else:
            hammer_pos = np.zeros(3)

        # Distance-based reward: reach towards hammer
        hand_to_hammer_dist = np.linalg.norm(right_hand_pos - hammer_pos)
        distance_reward = -hand_to_hammer_dist

        # Contact reward: bonus for maintaining contact with hammer
        contact_reward = 0.0
        if self.contact_with_hammer:
            self.grasp_contact_frames += 1
            contact_reward = 0.5  # Reward for contact
        else:
            self.grasp_contact_frames = 0

        # Grasp success: reward for stable grasp
        grasp_reward = 0.0
        if self.grasp_contact_frames > 10:  # Contact for multiple frames
            grasp_reward = 1.0

        # Action smoothness penalty
        action_diff = np.linalg.norm(action - self.last_action) if self.last_action is not None else 0.0
        smoothness_penalty = -self.smoothness_weight * action_diff * 0.01

        self.last_action = action.copy()

        # Combine rewards
        reward = distance_reward * 0.5 + contact_reward * 0.3 + grasp_reward * 0.2 + smoothness_penalty

        return float(reward)

    def render(self, mode='human') -> None:
        """Render the environment (if viewer available)."""
        if self.viewer is None:
            try:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print(f"Could not launch viewer: {e}")
                return

        if self.viewer is not None:
            self.viewer.sync()

    def close(self) -> None:
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()

    @property
    def observation_space(self):
        """Get the observation space shape."""
        # Right hand pos (3) + left hand pos (3) + hammer pos (3) + hammer vel (3) +
        # arm qpos (14) + arm qvel (14) = 40 dimensions
        return np.zeros(40)

    @property
    def action_space(self):
        """Get the action space shape."""
        return np.zeros(self.n_actions)


if __name__ == "__main__":
    # Test the environment
    print("Creating hammer grasp environment...")
    env = HammerGraspEnv()

    print("\nResetting environment...")
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space size: {env.n_actions}")

    print("\nRunning 100 steps with random actions...")
    for step in range(100):
        action = np.random.uniform(-1, 1, env.n_actions)
        obs, reward, done, info = env.step(action)

        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.4f}, contact={info['contact_with_hammer']}")

        if done:
            print(f"Episode done at step {step}")
            break

    print("\nEnvironment test complete!")
    env.close()
