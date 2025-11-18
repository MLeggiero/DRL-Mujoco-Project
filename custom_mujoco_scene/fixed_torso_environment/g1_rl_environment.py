#!/usr/bin/env python3
"""
Unitree G1 Reinforcement Learning Environment for Reach and Touch Tasks
"""
import mujoco
import numpy as np
import time
from typing import Dict, Tuple, Optional
import os

class G1ReachTouchEnv:
    def __init__(self, scene_path="../unitree_g1/g1_table_box_scene.xml"):
        """Initialize the G1 reach and touch environment"""

        self.scene_path = scene_path
        print(f"Using scene file: {self.scene_path}")

        # Check if scene file exists
        if not os.path.exists(self.scene_path):
            raise FileNotFoundError(f"Scene file not found: {self.scene_path}")

        self.model = None
        self.data = None
        self.viewer = None

        # Environment parameters
        # 400 steps with 0.3 action scaling (balanced speed and control)
        self.max_episode_steps = 400
        self.current_step = 0

        # Robot configuration
        self.initial_robot_pos = [-0.1, 0.0, 0.8]

        # Target objects (matching XML file)
        self.target_objects = ['red_box'] 
        self.current_target = None

        # Reward parameters
        self.success_distance = 0.05  # How close to consider "touching" the target (3 cm)
        self.last_distance = None
        self.last_action = None  # Track previous action for smoothness penalty

        # Reward scaling parameters
        self.typical_start_distance = 0.7  # Typical distance at episode start (meters)

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
        """Load the MuJoCo model"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.scene_path)
            self.data = mujoco.MjData(self.model)
            print(f"Loaded model: {self.model.nbody} bodies, {self.model.njnt} joints, {self.model.nu} actuators")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _setup_actuators(self):
        """Identify and map G1 actuators for right arm and torso control only"""
        self.controllable_actuators = []
        self.leg_actuators = []
        self.leg_joint_ids = []  # Track leg joint IDs for position locking
        self.initial_leg_qpos = {}  # Store initial leg positions

        # Find right arm and torso actuators, identify legs to lock
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

        # IMPORTANT: Map joint indices for observations (so we can observe what we control!)
        self._map_joint_indices()

    def _map_joint_indices(self):
        """Map joint indices for right arm and torso to include in observations"""
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
        """Reset the environment to initial state"""
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

        # Get initial observation
        obs = self._get_observation()

        print(f"Episode reset - Target: {self.current_target}")
        return obs
    
    def _reset_robot_pose(self):
        """Reset G1 to initial standing pose with locked legs"""
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
                
                # LEG JOINTS - Lock in stable standing position
                if 'hip' in joint_lower:
                    if 'pitch' in joint_lower or 'y' in joint_lower:
                        self.data.qpos[qpos_addr] = -0.15  # Hip pitch for standing
                    else:
                        self.data.qpos[qpos_addr] = 0.0
                elif 'knee' in joint_lower:
                    self.data.qpos[qpos_addr] = 0.2  # Slight knee bend for stability
                elif 'ankle' in joint_lower:
                    self.data.qpos[qpos_addr] = 0.0  # Neutral ankle
                
                # RIGHT ARM JOINTS - Set to reaching-ready position
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
                
                # LEFT ARM - Keep at side (neutral)
                elif 'left' in joint_lower and 'arm' in joint_lower:
                    self.data.qpos[qpos_addr] = 0.0
                
                # TORSO - Neutral
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
        """Add significant random variations to object positions for generalization"""
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
        """Execute one environment step"""
        # Clip action to reasonable range
        action = np.clip(action, -1, 1)

        # Apply action to robot actuators
        self._apply_action(action)

        # Step simulation multiple times for stability
        for _ in range(5):  # 5 simulation steps per RL step
            # Lock BEFORE step: floating base and legs
            if self.floating_base_qpos_addr is not None and self.initial_floating_base_qpos is not None:
                self.data.qpos[self.floating_base_qpos_addr:self.floating_base_qpos_addr+7] = self.initial_floating_base_qpos
                dof_start = 18  # Floating base starts at DOF 18 (from debug output)
                self.data.qvel[dof_start:dof_start+6] = 0.0

            # Lock leg joints before step
            for joint_id, initial_pos in self.initial_leg_qpos.items():
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                self.data.qpos[qpos_addr] = initial_pos
                self.data.qvel[qvel_addr] = 0.0

            # Step physics
            mujoco.mj_step(self.model, self.data)

            # Lock AFTER step: re-enforce constraints to counteract physics
            if self.floating_base_qpos_addr is not None and self.initial_floating_base_qpos is not None:
                self.data.qpos[self.floating_base_qpos_addr:self.floating_base_qpos_addr+7] = self.initial_floating_base_qpos
                dof_start = 18
                self.data.qvel[dof_start:dof_start+6] = 0.0

            # Lock leg joints after step (critical for preventing flailing)
            for joint_id, initial_pos in self.initial_leg_qpos.items():
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                self.data.qpos[qpos_addr] = initial_pos
                self.data.qvel[qvel_addr] = 0.0
        
        # Get observation
        obs = self._get_observation()

        # Calculate reward (pass action for smoothness penalty)
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
        """Apply RL action to right arm and torso actuators only"""

        # Clip actions to prevent wild movements
        action = np.clip(action, -1.0, 1.0)

        # Apply action only to controllable actuators (right arm + torso)
        # Higher torque scaling to encourage full arm extension (including elbow)
        for i, actuator_id in enumerate(self.controllable_actuators):
            if i < len(action):
                self.data.ctrl[actuator_id] = action[i] * 0.4  # Strong enough for elbow

        # IMPORTANT: Set leg actuator controls to ZERO
        # Legs should be completely still - no torques applied
        # The fixed base position handles stability, not actuator torques
        for leg_actuator_id in self.leg_actuators:
            self.data.ctrl[leg_actuator_id] = 0.0
    
    def _get_observation(self) -> Dict:
        """Get current observation (robot state + task info) - IMPROVED TO INCLUDE RIGHT ARM"""
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
        """Get position of current target object"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.current_target)
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return np.zeros(3)
    
    def _get_end_effector_position(self) -> np.ndarray:
        """Get position of robot's right arm end effector (hand/wrist) - IMPROVED"""
        # Cache the end effector ID after first successful lookup
        if not hasattr(self, '_end_effector_id'):
            self._end_effector_id = None

            # Try to find RIGHT hand/wrist SITE first (sites are more accurate)
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
        """Calculate distance from end effector to target"""
        ee_pos = self._get_end_effector_position()
        target_pos = self._get_target_position()
        return np.linalg.norm(ee_pos - target_pos)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward - HEAVILY WEIGHTED TOWARD VERY CLOSE PROXIMITY"""
        distance = self._get_distance_to_target()

        # 1. Exponential distance reward (gets much stronger when very close)
        # Using exponential: e^(-distance) heavily rewards being close
        distance_reward = -10.0 * distance  # Linear component for far distances

        # 2. Progress reward (only when very close to encourage final approach)
        progress_reward = 0.0
        if self.last_distance is not None:
            progress = self.last_distance - distance
            # Only reward progress when already somewhat close
            if distance < 0.2 and progress > 0:
                progress_reward = progress * 20.0  # Strong reward only when close
            elif progress < 0:
                progress_reward = progress * 2.0  # Penalize moving away

        # 3. Proximity bonuses - HEAVILY weighted toward very close distances
        proximity_bonus = 0.0
        if distance < 0.15:
            proximity_bonus += 5.0  # Small bonus at 15cm
        if distance < 0.10:
            proximity_bonus += 20.0  # Bigger bonus at 10cm
        if distance < 0.08:
            proximity_bonus += 50.0  # Large bonus at 8cm
        if distance < 0.06:
            proximity_bonus += 100.0  # Huge bonus at 6cm
        if distance < self.success_distance:
            proximity_bonus += 300.0  # MASSIVE success reward!
            print(f"Success! Reached {self.current_target}")

        # 4. Small action penalty
        action_penalty = -0.001 * np.sum(np.abs(action))

        total_reward = distance_reward + progress_reward + proximity_bonus + action_penalty

        # Update tracking
        self.last_distance = distance
        self.last_action = action.copy()

        return total_reward
    
    def _check_success(self) -> bool:
        """Check if task was completed successfully"""
        return self._get_distance_to_target() < self.success_distance
    
    def _check_done(self) -> bool:
        """Check if episode should end"""
        return (self.current_step >= self.max_episode_steps or 
                self._check_success())
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human' and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def close(self):
        """Close the environment"""
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