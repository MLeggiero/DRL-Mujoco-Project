#!/usr/bin/env python3
"""
Unitree G1 Reinforcement Learning Environment for Reaching Tasks.

This module implements a MuJoCo-based RL environment for training a Unitree G1
humanoid robot to perform reaching tasks with its right arm while maintaining
a stable base.
"""
import mujoco
import numpy as np
import time
from typing import Dict, Tuple, Optional
import os

class G1ReachTouchEnv:
    """MuJoCo environment for Unitree G1 reaching tasks.

    Implements a robot reaching task where the G1 humanoid uses its right arm
    to reach target objects while maintaining balance with locked legs.
    """

    def __init__(self, scene_path="../unitree_g1/g1_table_box_scene.xml",
                 action_smoothing=0.3, smoothness_weight=1.0, action_scale=0.25, sim_substeps=10):
        """Initialize the environment.

        Args:
            scene_path: Path to MuJoCo XML scene file
            action_smoothing: EMA coefficient for action filtering (0=max smoothing, 1=no filtering)
            smoothness_weight: Weight for action smoothness penalty in reward function
            action_scale: Scaling factor for actuator commands
            sim_substeps: Number of physics simulation steps per environment step
        """
        self.scene_path = scene_path
        print(f"Using scene file: {self.scene_path}")

        if not os.path.exists(self.scene_path):
            raise FileNotFoundError(f"Scene file not found: {self.scene_path}")

        self.model = None
        self.data = None
        self.viewer = None

        # Episode parameters
        self.max_episode_steps = 400
        self.current_step = 0

        # Robot configuration
        self.initial_robot_pos = [-0.1, 0.0, 0.8]

        # Target configuration
        self.target_objects = ['red_box']
        self.current_target = None

        # Action smoothing configuration
        self.action_smoothing = action_smoothing
        self.smoothness_weight = smoothness_weight
        self.action_scale = action_scale
        self.sim_substeps = sim_substeps
        self.filtered_action = None

        # Reward tracking
        self.success_distance = 0.05
        self.last_distance = None
        self.last_action = None
        self.typical_start_distance = 0.7

        print(f"Action smoothing: alpha={action_smoothing}, weight={smoothness_weight}")
        print(f"Action scale: {action_scale}, Substeps: {sim_substeps}")

        # Observation space indices (will be set after model loads)
        self.right_arm_qpos_indices = []
        self.torso_qpos_indices = []
        self.right_arm_qvel_indices = []
        self.torso_qvel_indices = []

        # Floating base stabilization (will be set after reset)
        self.floating_base_qpos_addr = None
        self.initial_floating_base_qpos = None

        # Vision parameters
        self.camera_width = 640
        self.camera_height = 480
        self.render_vision = False
        
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
        """Identify and map actuators for right arm and torso control.

        Separates controllable actuators (right arm and torso) from leg actuators,
        which are locked to maintain a stable base during reaching tasks.
        """
        self.controllable_actuators = []
        self.leg_actuators = []
        self.leg_joint_ids = []
        self.initial_leg_qpos = {}

        # Identify right arm and torso actuators, lock legs
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                actuator_lower = actuator_name.lower()

                # Lock all leg actuators (don't add to controllable list)
                if any(leg_part in actuator_lower for leg_part in ['hip', 'knee', 'ankle', 'leg']):
                    self.leg_actuators.append(i)
                    # Get the joint this actuator controls
                    joint_id = self.model.actuator_trnid[i, 0]
                    if joint_id >= 0 and joint_id not in self.leg_joint_ids:
                        self.leg_joint_ids.append(joint_id)
                    continue

                # Control right arm actuators
                if 'right' in actuator_lower and any(arm_part in actuator_lower for arm_part in ['shoulder', 'elbow', 'wrist', 'arm']):
                    self.controllable_actuators.append(i)

                # Control torso/waist actuators (helps with reaching)
                elif any(torso_part in actuator_lower for torso_part in ['torso', 'waist', 'spine']):
                    self.controllable_actuators.append(i)

        print(f"Controllable actuators (right arm + torso): {len(self.controllable_actuators)} actuators")
        print(f"Locked leg actuators: {len(self.leg_actuators)} actuators")
        print(f"Locked leg joints: {len(self.leg_joint_ids)} joints")

        # Get total controllable actuators (right arm + torso only)
        self.n_actions = len(self.controllable_actuators)
        if self.n_actions == 0:
            print("WARNING: No arm/torso actuators found - using first 6 actuators")
            self.n_actions = min(self.model.nu, 6)
            self.controllable_actuators = list(range(self.n_actions))

        # Map joint indices for observations
        self._map_joint_indices()

    def _map_joint_indices(self):
        """Map joint indices for right arm and torso to observation space.

        Identifies position and velocity indices for controllable joints,
        enabling the policy to observe the state of actuated degrees of freedom.
        """
        self.right_arm_qpos_indices = []
        self.torso_qpos_indices = []
        self.right_arm_qvel_indices = []
        self.torso_qvel_indices = []

        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_lower = joint_name.lower()

                # Find right arm joints
                if 'right' in joint_lower and any(part in joint_lower for part in ['shoulder', 'elbow', 'wrist', 'arm']):
                    if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                        qpos_addr = self.model.jnt_qposadr[i]
                        qvel_addr = self.model.jnt_dofadr[i]
                        self.right_arm_qpos_indices.append(qpos_addr)
                        self.right_arm_qvel_indices.append(qvel_addr)

                # Find torso joints
                elif any(part in joint_lower for part in ['torso', 'waist', 'spine']):
                    if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                        qpos_addr = self.model.jnt_qposadr[i]
                        qvel_addr = self.model.jnt_dofadr[i]
                        self.torso_qpos_indices.append(qpos_addr)
                        self.torso_qvel_indices.append(qvel_addr)

        print(f"Right arm qpos indices for observation: {self.right_arm_qpos_indices}")
        print(f"Right arm qvel indices for observation: {self.right_arm_qvel_indices}")
        print(f"Torso qpos indices for observation: {self.torso_qpos_indices}")
        print(f"Torso qvel indices for observation: {self.torso_qvel_indices}")

    def reset(self, target_object=None) -> Dict:
        """Reset the environment to initial state.

        Args:
            target_object: Name of target object, or None for random selection

        Returns:
            Initial observation dictionary
        """
        self.current_step = 0

        # Reset to initial configuration from XML
        mujoco.mj_resetData(self.model, self.data)

        # Reset robot to initial pose (after mj_resetData)
        self._reset_robot_pose()

        # Choose target object
        if target_object is None:
            self.current_target = np.random.choice(self.target_objects)
        else:
            self.current_target = target_object

        # Add small randomization to object positions (optional)
        self._randomize_object_positions()

        # Forward simulation to update state
        mujoco.mj_forward(self.model, self.data)

        # Reset reward tracking
        self.last_distance = None
        self.last_action = None
        self.filtered_action = None  # Reset action filter

        # Get initial observation
        obs = self._get_observation()

        print(f"Episode reset - Target: {self.current_target}")
        return obs
    
    def _reset_robot_pose(self):
        """Reset robot to initial standing pose with locked legs.

        Configures the humanoid in a stable standing position with legs locked
        and right arm positioned for reaching tasks.
        """
        # Reset all velocities
        self.data.qvel[:] = 0
        
        # Set robot position
        for i in range(self.model.njnt):
            joint_type = self.model.jnt_type[i]
            qpos_addr = self.model.jnt_qposadr[i]
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            joint_lower = joint_name.lower()
            
            if joint_type == mujoco.mjtJoint.mjJNT_FREE and qpos_addr + 6 < self.model.nq:
                # Only modify robot's floating base, not object free joints
                if 'floating_base' in joint_lower or 'base' in joint_lower or 'pelvis' in joint_lower:
                    # Robot floating base: set to standing position
                    self.data.qpos[qpos_addr:qpos_addr+3] = self.initial_robot_pos
                    self.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]  # Quaternion

                    # Save floating base info
                    self.floating_base_qpos_addr = qpos_addr
                    self.initial_floating_base_qpos = self.data.qpos[qpos_addr:qpos_addr+7].copy()
                # Skip object free joints - they were already set correctly by mj_resetData
                
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE and qpos_addr < self.model.nq:
                # Set joint positions - legs locked, arms movable

                # Leg joints - lock in stable standing position
                if 'hip' in joint_lower:
                    if 'pitch' in joint_lower or 'y' in joint_lower:
                        self.data.qpos[qpos_addr] = -0.15  # Hip pitch for standing
                    else:
                        self.data.qpos[qpos_addr] = 0.0
                elif 'knee' in joint_lower:
                    self.data.qpos[qpos_addr] = 0.2  # Slight knee bend for stability
                elif 'ankle' in joint_lower:
                    self.data.qpos[qpos_addr] = 0.0  # Neutral ankle

                # Right arm joints - set to reaching-ready position
                elif 'right' in joint_lower:
                    if 'shoulder' in joint_lower:
                        if 'pitch' in joint_lower or 'y' in joint_lower:
                            self.data.qpos[qpos_addr] = -0.5  # Arm forward
                        elif 'roll' in joint_lower or 'x' in joint_lower:
                            self.data.qpos[qpos_addr] = 0.2   # Arm slightly out
                        else:
                            self.data.qpos[qpos_addr] = 0.0
                    elif 'elbow' in joint_lower:
                        self.data.qpos[qpos_addr] = -0.8  # Bent elbow ready to reach
                    else:
                        self.data.qpos[qpos_addr] = 0.0

                # Left arm - keep at side (neutral)
                elif 'left' in joint_lower and 'arm' in joint_lower:
                    self.data.qpos[qpos_addr] = 0.0

                # Torso - neutral position
                elif any(part in joint_lower for part in ['torso', 'waist', 'spine']):
                    self.data.qpos[qpos_addr] = 0.0

                else:
                    self.data.qpos[qpos_addr] = 0.0

        # Store initial leg positions for locking during simulation
        self.initial_leg_qpos.clear()
        for joint_id in self.leg_joint_ids:
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.initial_leg_qpos[joint_id] = self.data.qpos[qpos_addr]
    
    def _randomize_object_positions(self):
        """Randomize target object positions to improve policy generalization.

        Positions are randomized within a reachable workspace region to encourage
        the policy to learn robust reaching behaviors.
        """
        for obj_name in self.target_objects:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id >= 0:
                # Find the free joint for this object
                for i in range(self.model.njnt):
                    if self.model.jnt_bodyid[i] == body_id and self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                        qpos_addr = self.model.jnt_qposadr[i]
                        if qpos_addr + 6 < self.model.nq:
                            # Randomize position in a reachable region
                            # X: 0.4m to 0.6m (forward distance)
                            # Y: -0.15m to +0.15m (left/right)
                            # Z: On table (0.74m)
                            x = np.random.uniform(0.35, 0.65)
                            y = np.random.uniform(-0.2, 0.2)
                            z = 0.74  # Table height
                            self.data.qpos[qpos_addr:qpos_addr+3] = [x, y, z]
                        break
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute one environment step with action smoothing.

        Args:
            action: Control commands for actuators

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Clip action to valid range
        action = np.clip(action, -1, 1)

        # Apply exponential moving average filter for smoothness
        if self.filtered_action is None:
            self.filtered_action = action.copy()
        else:
            # EMA: filtered = alpha * new + (1-alpha) * old (lower alpha = more smoothing)
            self.filtered_action = (self.action_smoothing * action +
                                   (1 - self.action_smoothing) * self.filtered_action)

        # Apply filtered action to robot actuators
        self._apply_action(self.filtered_action)

        # Step simulation multiple times for stability
        for _ in range(self.sim_substeps):
            # Lock floating base and legs before physics step
            if self.floating_base_qpos_addr is not None and self.initial_floating_base_qpos is not None:
                self.data.qpos[self.floating_base_qpos_addr:self.floating_base_qpos_addr+7] = self.initial_floating_base_qpos
                dof_start = 18
                self.data.qvel[dof_start:dof_start+6] = 0.0

            # Lock leg joints before step
            for joint_id, initial_pos in self.initial_leg_qpos.items():
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                self.data.qpos[qpos_addr] = initial_pos
                self.data.qvel[qvel_addr] = 0.0

            # Step physics
            mujoco.mj_step(self.model, self.data)

            # Re-enforce constraints after physics step
            if self.floating_base_qpos_addr is not None and self.initial_floating_base_qpos is not None:
                self.data.qpos[self.floating_base_qpos_addr:self.floating_base_qpos_addr+7] = self.initial_floating_base_qpos
                dof_start = 18
                self.data.qvel[dof_start:dof_start+6] = 0.0

            # Lock leg joints after step
            for joint_id, initial_pos in self.initial_leg_qpos.items():
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                self.data.qpos[qpos_addr] = initial_pos
                self.data.qvel[qvel_addr] = 0.0

        # Get observation
        obs = self._get_observation()

        # Calculate reward (pass raw action for smoothness penalty)
        reward = self._calculate_reward(action)

        # Check if episode is done
        done = self._check_done()

        # Additional info
        info = {
            'target': self.current_target,
            'distance_to_target': self._get_distance_to_target(),
            'success': self._check_success()
        }

        self.current_step += 1

        return obs, reward, done, info
    
    def _apply_action(self, action):
        """Apply action to right arm and torso actuators.

        Args:
            action: Clipped action vector for controllable actuators
        """
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)

        # Apply scaled action to controllable actuators
        for i, actuator_id in enumerate(self.controllable_actuators):
            if i < len(action):
                self.data.ctrl[actuator_id] = action[i] * self.action_scale

        # Set leg actuator controls to zero (legs are locked via position constraints)
        for leg_actuator_id in self.leg_actuators:
            self.data.ctrl[leg_actuator_id] = 0.0
    
    def _get_observation(self) -> Dict:
        """Get current observation including robot state and task information.

        Returns:
            Dictionary containing joint states, end-effector position,
            target position, and derived task-relevant features
        """
        obs = {}

        # Collect relevant joint positions and velocities (right arm + torso only)
        qpos_indices = self.right_arm_qpos_indices + self.torso_qpos_indices
        qvel_indices = self.right_arm_qvel_indices + self.torso_qvel_indices

        if len(qpos_indices) > 0 and len(qvel_indices) > 0:
            # Get positions for controllable joints
            obs['robot_qpos'] = self.data.qpos[qpos_indices].copy()
            # Get velocities for controllable joints (using proper qvel indices)
            obs['robot_qvel'] = self.data.qvel[qvel_indices].copy()
        else:
            # Fallback: if joint mapping failed, use first 10 joints
            print("Warning: Using fallback observation (first 10 joints)")
            obs['robot_qpos'] = self.data.qpos[:10].copy()
            obs['robot_qvel'] = self.data.qvel[:10].copy()

        # End-effector position (3D)
        obs['end_effector_pos'] = self._get_end_effector_position()

        # Target object position (3D)
        target_pos = self._get_target_position()
        obs['target_position'] = target_pos

        # Relative vector from hand to target (3D) - very important for learning!
        obs['hand_to_target'] = target_pos - obs['end_effector_pos']

        # Distance to target (scalar)
        obs['distance_to_target'] = self._get_distance_to_target()

        return obs
    
    def _get_target_position(self) -> np.ndarray:
        """Get position of current target object.

        Returns:
            3D position vector of target object
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.current_target)
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return np.zeros(3)

    def _get_end_effector_position(self) -> np.ndarray:
        """Get position of robot's right arm end effector.

        Returns:
            3D position vector of end effector (hand/wrist)
        """
        # Cache the end effector ID after first successful lookup
        if not hasattr(self, '_end_effector_id'):
            self._end_effector_id = None

            # Try to find right hand/wrist site first (sites are more accurate)
            right_hand_site_names = [
                'right_palm', 'right_hand_site', 'right_end_effector',
                'right_wrist_site', 'r_palm', 'palm_right'
            ]

            for site_name in right_hand_site_names:
                try:
                    site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    if site_id >= 0:
                        self._end_effector_id = ('site', site_id)
                        print(f"Using end effector SITE: {site_name} (id={site_id})")
                        break
                except:
                    pass

            # If no site found, try bodies
            if self._end_effector_id is None:
                right_hand_names = [
                    'right_hand', 'right_wrist', 'hand_right', 'wrist_right',
                    'r_hand', 'r_wrist', 'right_palm', 'hand_r', 'wrist_r'
                ]

                for hand_name in right_hand_names:
                    hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_name)
                    if hand_id >= 0:
                        self._end_effector_id = ('body', hand_id)
                        print(f"Using end effector BODY: {hand_name} (id={hand_id})")
                        break

            # Fallback: search for any body/site with 'right' and 'hand'/'wrist'/'palm'
            if self._end_effector_id is None:
                print("WARNING: Searching for right arm end effector...")
                for body_id in range(self.model.nbody):
                    body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    if body_name:
                        body_lower = body_name.lower()
                        if 'right' in body_lower and any(kw in body_lower for kw in ['hand', 'wrist', 'palm']):
                            self._end_effector_id = ('body', body_id)
                            print(f"Found fallback end effector: {body_name} (body_id={body_id})")
                            break

            # Final fallback
            if self._end_effector_id is None:
                print("WARNING: Could not find right hand - using fallback body index")
                fallback_id = min(15, self.model.nbody - 1)
                self._end_effector_id = ('body', fallback_id)

        # Return position based on cached ID
        obj_type, obj_id = self._end_effector_id
        if obj_type == 'site':
            return self.data.site_xpos[obj_id].copy()
        else:
            return self.data.xpos[obj_id].copy()
    
    def _get_distance_to_target(self) -> float:
        """Calculate Euclidean distance from end effector to target.

        Returns:
            Distance in meters
        """
        ee_pos = self._get_end_effector_position()
        target_pos = self._get_target_position()
        return np.linalg.norm(ee_pos - target_pos)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward function for reaching task.

        Combines distance-based rewards, progress incentives, proximity bonuses,
        and smoothness penalties to encourage efficient reaching behavior.

        Args:
            action: Raw action vector for smoothness penalty computation

        Returns:
            Scalar reward value
        """
        distance = self._get_distance_to_target()

        # Distance reward with exponential shaping for better gradient
        # Steeper exponential decay (-10.0) ensures rewards stay negative until very close to goal
        distance_reward = -5.0 * distance + 10.0 * np.exp(-10.0 * distance)

        # Progress reward (active throughout episode)
        progress_reward = 0.0
        if self.last_distance is not None:
            progress = self.last_distance - distance
            progress_reward = progress * 50.0
            # Triple penalty for moving away from target
            if progress < 0:
                progress_reward *= 3.0

        # Proximity bonuses for approaching target
        proximity_bonus = 0.0
        if distance < 0.20:
            proximity_bonus += 10.0
        if distance < 0.15:
            proximity_bonus += 20.0
        if distance < 0.10:
            proximity_bonus += 50.0
        if distance < 0.08:
            proximity_bonus += 100.0
        if distance < 0.06:
            proximity_bonus += 200.0
        if distance < self.success_distance:
            proximity_bonus += 2000.0
            print(f"Success! Reached {self.current_target}")

        # Velocity penalty to encourage smooth joint movements
        velocity_penalty = 0.0
        qvel_indices = self.right_arm_qvel_indices + self.torso_qvel_indices
        if len(qvel_indices) > 0:
            joint_velocities = self.data.qvel[qvel_indices]
            velocity_penalty = -0.01 * np.sum(np.square(joint_velocities))

        # Action magnitude penalty for efficiency
        action_penalty = -0.005 * np.sum(np.square(action))

        # Smoothness penalty to discourage jerky movements
        smoothness_penalty = 0.0
        if self.last_action is not None:
            action_change = np.sum(np.square(action - self.last_action))
            smoothness_penalty = -self.smoothness_weight * action_change

        total_reward = (distance_reward + progress_reward + proximity_bonus +
                       velocity_penalty + action_penalty + smoothness_penalty)

        # Update tracking
        self.last_distance = distance
        self.last_action = action.copy()

        return total_reward

    def _check_success(self) -> bool:
        """Check if task was completed successfully.

        Returns:
            True if end effector is within success threshold of target
        """
        return self._get_distance_to_target() < self.success_distance

    def _check_done(self) -> bool:
        """Check if episode should terminate.

        Returns:
            True if episode reached max steps or task succeeded
        """
        return (self.current_step >= self.max_episode_steps or
                self._check_success())

    def render(self, mode='human'):
        """Render the environment.

        Args:
            mode: Rendering mode ('human' for interactive viewer)
        """
        if mode == 'human' and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close(self):
        """Close the environment and release resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def test_environment():
    """Test the RL environment"""
    print("Testing G1 Reach-Touch RL Environment")
    print("=" * 50)
    
    # Check current directory
    print(f"Current directory: {os.getcwd()}")
    
    # Try to find scene file
    possible_paths = [
        "unitree_g1/g1_table_box_scene.xml",
        "g1_table_box_scene.xml",
        "../unitree_g1/g1_table_box_scene.xml",
    ]
    
    scene_found = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found scene file: {path}")
            scene_found = True
            break
    
    if not scene_found:
        print("ERROR: Scene file not found in any expected location!")
        print("\nSearched locations:")
        for path in possible_paths:
            full_path = os.path.abspath(path)
            exists = "EXISTS" if os.path.exists(path) else "NOT FOUND"
            print(f"  [{exists}] {path}")
            print(f"           -> {full_path}")
        
        print("\nExpected file structure:")
        print("  mujoco_projects/")
        print("    ├── unitree_g1/")
        print("    │   └── g1_table_box_scene.xml")
        print("    ├── mujoco_menagerie/")
        print("    │   └── unitree_g1/")
        print("    │       ├── g1.xml")
        print("    │       └── assets/")
        print("    ├── g1_rl_environment.py")
        print("    └── g1_training_script.py")
        
        return False
    
    try:
        # Create environment
        env = G1ReachTouchEnv()
        
        # Test reset
        obs = env.reset(target_object='red_box')
        print(f"Environment reset complete")
        print(f"  Target: {env.current_target}")
        print(f"  Distance to target: {obs['distance_to_target']:.3f}")
        print(f"  Action space size: {env.n_actions}")
        
        # Test random actions
        print("\nRunning random actions test...")
        for step in range(20):
            # Random action
            action = np.random.uniform(-0.5, 0.5, env.n_actions)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            if step % 5 == 0:
                print(f"  Step {step}: distance={obs['distance_to_target']:.3f}, reward={reward:.2f}")
            
            if done:
                print(f"  Episode ended at step {step}")
                if info['success']:
                    print("  Task completed successfully!")
                break
        
        print("Environment test completed successfully")
        env.close()
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_environment()