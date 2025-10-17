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
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Robot configuration
        self.initial_robot_pos = [-0.1, 0.0, 0.8]
        
        # Target objects (matching your XML file)
        self.target_objects = ['red_box', 'blue_cylinder', 'blue_cylinder2', 'green_cone']
        self.current_target = None
        
        # Reward parameters
        self.success_distance = 0.08  # How close to consider "touching"
        self.last_distance = None
        
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
        
        # Find right arm and torso actuators, identify legs to lock
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                actuator_lower = actuator_name.lower()
                
                # Lock all leg actuators (don't add to controllable list)
                if any(leg_part in actuator_lower for leg_part in ['hip', 'knee', 'ankle', 'leg']):
                    self.leg_actuators.append(i)
                    continue
                
                # Control right arm actuators
                if 'right' in actuator_lower and any(arm_part in actuator_lower for arm_part in ['shoulder', 'elbow', 'wrist', 'arm']):
                    self.controllable_actuators.append(i)
                    
                # Control torso/waist actuators (helps with reaching)
                elif any(torso_part in actuator_lower for torso_part in ['torso', 'waist', 'spine']):
                    self.controllable_actuators.append(i)
        
        print(f"Controllable actuators (right arm + torso): {len(self.controllable_actuators)} actuators")
        print(f"Locked leg actuators: {len(self.leg_actuators)} actuators")
        
        # Get total controllable actuators (right arm + torso only)
        self.n_actions = len(self.controllable_actuators)
        if self.n_actions == 0:
            print("WARNING: No arm/torso actuators found - using first 6 actuators")
            self.n_actions = min(self.model.nu, 6)
            self.controllable_actuators = list(range(self.n_actions))
    
    def reset(self, target_object=None) -> Dict:
        """Reset the environment to initial state"""
        self.current_step = 0
        
        # Reset robot to initial pose
        self._reset_robot_pose()
        
        # Choose target object
        if target_object is None:
            self.current_target = np.random.choice(self.target_objects)
        else:
            self.current_target = target_object
            
        # Reset object positions (add some randomization)
        self._randomize_object_positions()
        
        # Forward simulation to update state
        mujoco.mj_forward(self.model, self.data)
        
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
                # Free joint: standing position
                self.data.qpos[qpos_addr:qpos_addr+3] = self.initial_robot_pos
                self.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]  # Quaternion
                
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
    
    def _randomize_object_positions(self):
        """Add small random variations to object positions"""
        for obj_name in self.target_objects:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id >= 0:
                # Find the free joint for this object
                for i in range(self.model.njnt):
                    if self.model.jnt_bodyid[i] == body_id and self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                        qpos_addr = self.model.jnt_qposadr[i]
                        if qpos_addr + 6 < self.model.nq:
                            # Add small random offset (+/-3cm)
                            original_pos = self.data.qpos[qpos_addr:qpos_addr+3].copy()
                            noise = np.random.uniform(-0.03, 0.03, 3)
                            noise[2] = max(0, noise[2])  # Don't go below table
                            self.data.qpos[qpos_addr:qpos_addr+3] = original_pos + noise
                        break
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute one environment step"""
        # Clip action to reasonable range
        action = np.clip(action, -1, 1)
        
        # Apply action to robot actuators
        self._apply_action(action)
        
        # Step simulation multiple times for stability
        for _ in range(5):  # 5 simulation steps per RL step
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
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
        
        # Apply action only to controllable actuators (right arm + torso)
        for i, actuator_id in enumerate(self.controllable_actuators):
            if i < len(action):
                self.data.ctrl[actuator_id] = action[i] * 10.0  # Scale torque
        
        # Keep leg actuators at zero (locked in standing position)
        for leg_actuator_id in self.leg_actuators:
            self.data.ctrl[leg_actuator_id] = 0.0
    
    def _get_observation(self) -> Dict:
        """Get current observation (robot state + task info)"""
        obs = {}
        
        # Robot proprioceptive state (limited to prevent huge observation space)
        obs['robot_qpos'] = self.data.qpos[:min(20, self.model.nq)].copy()
        obs['robot_qvel'] = self.data.qvel[:min(20, self.model.nv)].copy()
        
        # Target object position
        target_pos = self._get_target_position()
        obs['target_position'] = target_pos
        
        # End-effector position
        obs['end_effector_pos'] = self._get_end_effector_position()
        
        # Distance to target
        obs['distance_to_target'] = self._get_distance_to_target()
        
        return obs
    
    def _get_target_position(self) -> np.ndarray:
        """Get position of current target object"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.current_target)
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return np.zeros(3)
    
    def _get_end_effector_position(self) -> np.ndarray:
        """Get position of robot's right arm end effector (hand/wrist)"""
        # Try to find RIGHT hand/wrist body specifically
        right_hand_names = [
            'right_hand', 'right_wrist', 'hand_right', 'wrist_right', 
            'r_hand', 'r_wrist', 'right_end_effector', 'right_palm',
            'hand_r', 'wrist_r'
        ]
        
        for hand_name in right_hand_names:
            hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_name)
            if hand_id >= 0:
                # Only print this once during initialization
                if not hasattr(self, '_end_effector_logged'):
                    print(f"Using end effector: {hand_name} (body_id={hand_id})")
                    self._end_effector_logged = True
                return self.data.xpos[hand_id].copy()
        
        # Fallback: search for any body with 'right' and 'hand' or 'wrist'
        if not hasattr(self, '_end_effector_search_logged'):
            print("Searching for right arm end effector in body names...")
            self._end_effector_search_logged = True
            
        for body_id in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name:
                body_lower = body_name.lower()
                if 'right' in body_lower and ('hand' in body_lower or 'wrist' in body_lower):
                    if not hasattr(self, '_end_effector_fallback_logged'):
                        print(f"Found right arm end effector: {body_name} (body_id={body_id})")
                        self._end_effector_fallback_logged = True
                    return self.data.xpos[body_id].copy()
        
        # Last resort: use a body index that's likely to be near the arm
        if not hasattr(self, '_end_effector_warning_logged'):
            print("WARNING: Could not find right hand/wrist - using approximate body index")
            self._end_effector_warning_logged = True
            
        if self.model.nbody > 15:
            return self.data.xpos[15].copy()
        elif self.model.nbody > 10:
            return self.data.xpos[10].copy()
        
        return np.zeros(3)
    
    def _get_distance_to_target(self) -> float:
        """Calculate distance from end effector to target"""
        ee_pos = self._get_end_effector_position()
        target_pos = self._get_target_position()
        return np.linalg.norm(ee_pos - target_pos)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        distance = self._get_distance_to_target()
        
        # Distance-based reward (closer = better)
        distance_reward = -distance * 10.0
        
        # Progress reward (getting closer)
        progress_reward = 0.0
        if self.last_distance is not None:
            progress = self.last_distance - distance
            progress_reward = progress * 50.0  # Scale progress
        
        # Success reward
        success_reward = 0.0
        if distance < self.success_distance:
            success_reward = 100.0
            print(f"Success! Reached {self.current_target}")
        
        # Penalty for time (encourage efficiency)
        time_penalty = -0.1
        
        total_reward = distance_reward + progress_reward + success_reward + time_penalty
        self.last_distance = distance
        
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