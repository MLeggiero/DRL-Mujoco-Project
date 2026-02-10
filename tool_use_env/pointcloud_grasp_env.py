#!/usr/bin/env python3
"""
RL Environment for learning grasps from point cloud observations.

This environment:
- Provides point cloud observations of the scene
- Rewards successful grasps based on contact and lift
- Uses continuous actions for 6-DOF gripper control
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from camera_utils import CameraProcessor
import cv2


class PointCloudGraspEnv(gym.Env):
    """
    Gymnasium environment for learning grasping from point clouds.

    Observation Space:
        - Point cloud (downsampled to fixed size)
        - Current gripper position and orientation
        - Gripper width

    Action Space:
        - Delta gripper position (3D)
        - Delta gripper orientation (3D rotation)
        - Gripper open/close command
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        scene_path="hammer_grasp_rgbd_scene.xml",
        camera_name="track_front",
        max_episode_steps=200,
        point_cloud_size=1024,  # Downsample to this many points
        action_scale=0.02,  # 2cm per action step
        success_lift_height=0.05,  # 5cm lift = success
        settle_steps=100,
        image_width=640,
        image_height=480,
        render_mode=None
    ):
        super().__init__()

        self.scene_path = Path(__file__).parent / scene_path
        self.camera_name = camera_name
        self.max_episode_steps = max_episode_steps
        self.point_cloud_size = point_cloud_size
        self.action_scale = action_scale
        self.success_lift_height = success_lift_height
        self.settle_steps = settle_steps
        self.image_width = image_width
        self.image_height = image_height
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        # Camera processor
        self.camera_processor = CameraProcessor(
            self.model,
            width=image_width,
            height=image_height
        )

        # Renderer
        self.renderer = mujoco.Renderer(self.model, height=image_height, width=image_width)

        # Get important body/site IDs
        self.hammer_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hammer")
        self.right_hand_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")

        # Define observation space
        # Point cloud: N points × 6 (xyz + rgb)
        # Proprioception: 7 (gripper pos + quat)
        obs_dim = point_cloud_size * 6 + 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Define action space
        # [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.hammer_initial_height = None
        self.episode_contacts = 0

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)

        # Let scene settle
        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)

        # Record initial hammer height
        self.hammer_initial_height = self.data.xpos[self.hammer_body_id][2]

        # Reset tracking
        self.current_step = 0
        self.episode_contacts = 0

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one environment step."""
        # Apply action to robot
        self._apply_action(action)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation and reward
        obs = self._get_observation()
        reward = self._compute_reward()
        info = self._get_info()

        # Check termination
        self.current_step += 1
        terminated = info['success']
        truncated = self.current_step >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current observation (point cloud + proprioception)."""
        # Capture RGB-D
        rgb, depth = self._capture_rgbd()

        # Generate point cloud
        points, colors = self.camera_processor.rgbd_to_pointcloud(
            rgb, depth, self.camera_name,
            min_depth=0.1, max_depth=3.0
        )

        # Downsample to fixed size
        if len(points) > self.point_cloud_size:
            indices = np.random.choice(len(points), self.point_cloud_size, replace=False)
            points = points[indices]
            colors = colors[indices]
        elif len(points) < self.point_cloud_size:
            # Pad with zeros if not enough points
            pad_size = self.point_cloud_size - len(points)
            points = np.vstack([points, np.zeros((pad_size, 3))])
            colors = np.vstack([colors, np.zeros((pad_size, 3))])

        # Normalize point cloud (center around origin)
        points_normalized = points - points.mean(axis=0)
        colors_normalized = colors / 255.0

        # Combine points and colors
        point_cloud = np.hstack([points_normalized, colors_normalized]).flatten()

        # Get gripper proprioception
        gripper_pos = self.data.site_xpos[self.right_hand_site_id]
        gripper_quat = np.array([1, 0, 0, 0])  # Simplified - extract from rotation matrix if needed
        proprioception = np.hstack([gripper_pos, gripper_quat])

        # Combine into observation
        obs = np.hstack([point_cloud, proprioception]).astype(np.float32)

        return obs

    def _capture_rgbd(self):
        """Capture RGB-D from camera."""
        # RGB
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.renderer.render()

        # Depth
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_name)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        return rgb, depth

    def _apply_action(self, action):
        """
        Apply action to robot gripper.

        Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Scale position deltas
        delta_pos = action[:3] * self.action_scale

        # Get current gripper position
        current_pos = self.data.site_xpos[self.right_hand_site_id].copy()

        # Compute target position
        target_pos = current_pos + delta_pos

        # Use inverse kinematics to reach target (simplified - direct joint control)
        # In real implementation, use IK solver from motion_planner.py

        # For now, use direct joint control for arm
        # This is simplified - you should use proper IK
        arm_joint_ids = self._get_arm_joint_ids()

        # Simple proportional control toward target
        for joint_id in arm_joint_ids:
            # Apply small random movements (placeholder for IK)
            self.data.ctrl[joint_id] = np.clip(
                self.data.qpos[joint_id] + np.random.randn() * 0.01,
                self.model.jnt_range[joint_id, 0],
                self.model.jnt_range[joint_id, 1]
            )

        # Gripper control (open/close)
        gripper_cmd = action[6]
        finger_joint_ids = self._get_finger_joint_ids()
        for joint_id in finger_joint_ids:
            if gripper_cmd > 0:  # Close
                self.data.ctrl[joint_id] = self.model.jnt_range[joint_id, 0]
            else:  # Open
                self.data.ctrl[joint_id] = self.model.jnt_range[joint_id, 1]

    def _compute_reward(self):
        """
        Compute reward based on grasping success.

        Reward components:
        1. Distance to hammer (sparse)
        2. Contact with hammer (+)
        3. Lift hammer above threshold (++)
        4. Drop hammer (-)
        """
        reward = 0.0

        # Get positions
        gripper_pos = self.data.site_xpos[self.right_hand_site_id]
        hammer_pos = self.data.xpos[self.hammer_body_id]

        # 1. Distance reward (negative, encourages approaching)
        distance = np.linalg.norm(gripper_pos - hammer_pos)
        reward -= distance * 0.1

        # 2. Contact reward
        contacts = self._check_hammer_contact()
        if contacts > 0:
            reward += 0.5
            self.episode_contacts += 1

        # 3. Lift reward (big bonus)
        hammer_height = hammer_pos[2]
        lift_height = hammer_height - self.hammer_initial_height

        if lift_height > self.success_lift_height:
            reward += 10.0  # Success!
        elif lift_height > 0.01:  # Small lift
            reward += lift_height * 10.0

        # 4. Penalty for dropping
        if self.episode_contacts > 10 and lift_height < 0.005:
            reward -= 1.0

        return reward

    def _check_hammer_contact(self):
        """Check if gripper is in contact with hammer."""
        num_contacts = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            if geom1 and geom2:
                if ('hammer' in geom1 and 'hand' in geom2) or \
                   ('hammer' in geom2 and 'hand' in geom1):
                    num_contacts += 1

        return num_contacts

    def _get_info(self):
        """Get auxiliary information."""
        hammer_pos = self.data.xpos[self.hammer_body_id]
        lift_height = hammer_pos[2] - self.hammer_initial_height

        return {
            'hammer_position': hammer_pos.copy(),
            'lift_height': lift_height,
            'success': lift_height > self.success_lift_height,
            'contacts': self.episode_contacts,
            'step': self.current_step
        }

    def _get_arm_joint_ids(self):
        """Get right arm joint IDs."""
        arm_joints = [
            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_pitch_joint',
            'right_wrist_yaw_joint',
            'right_wrist_roll_joint',
            'right_wrist_pitch_joint'
        ]
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in arm_joints if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0]

    def _get_finger_joint_ids(self):
        """Get right hand finger joint IDs."""
        finger_joints = [
            'right_hand_thumb_joint',
            'right_hand_index_joint',
            'right_hand_middle_joint'
        ]
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in finger_joints if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0]

    def render(self):
        """Render the environment."""
        if self.render_mode == 'rgb_array':
            rgb, _ = self._capture_rgbd()
            return rgb
        elif self.render_mode == 'human':
            # Use MuJoCo viewer
            pass

    def close(self):
        """Clean up resources."""
        pass


if __name__ == "__main__":
    # Test the environment
    env = PointCloudGraspEnv()

    print("Testing PointCloudGraspEnv...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    # Take random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Lift height: {info['lift_height']:.4f}m")
        print(f"  Contacts: {info['contacts']}")

        if terminated or truncated:
            print("Episode ended!")
            break

    env.close()
    print("\nEnvironment test complete!")
