#!/usr/bin/env python3
"""
RGBD Vision-Based Hammer Grasping Environment for Unitree G1.

This environment provides RGB-D camera input instead of proprioceptive state,
simulating realistic perception for the G1 robot's onboard cameras.

The robot receives:
- RGBD images from 2 front-facing cameras (stereo setup)
- Low-level proprioceptive feedback (joint angles, velocities)
- Action commands for arm/hand control

This is more realistic for actual robot deployment.
"""

import mujoco
import numpy as np
from typing import Dict, Tuple, Optional, Union
import os
from collections import deque


class RGBDCamera:
    """Simulated RGBD camera attached to robot."""

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 camera_name: str,
                 image_size: Tuple[int, int] = (320, 240),
                 depth_range: Tuple[float, float] = (0.01, 10.0)):
        """Initialize RGBD camera.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            camera_name: Name of camera in XML scene
            image_size: (width, height) of rendered images
            depth_range: (near, far) clipping planes
        """
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width, self.height = image_size
        self.depth_range = depth_range

        # Get camera ID
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        if self.camera_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")

        # Create rendering context
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()

        # Initialize camera from model
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamType.mjCAMERA_FIXED
        self.camera.fixedcamid = self.camera_id

    def render(self) -> Tuple[np.ndarray, np.ndarray]:
        """Render RGB and depth images from camera.

        Returns:
            Tuple of (rgb_image, depth_image)
            - rgb_image: (height, width, 3) uint8
            - depth_image: (height, width) float32, normalized to [0, 1]
        """
        # Update scene
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)

        # Render RGB
        rgb_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth_array = np.zeros((self.height, self.width), dtype=np.float32)

        mujoco.mjr_render(self.width, self.height, self.scene, self.context)

        # Read pixels
        mujoco.mjr_readPixels(rgb_array, depth_array, self.scene, self.context)

        # Normalize depth to [0, 1] range
        depth_array = np.clip(depth_array, self.depth_range[0], self.depth_range[1])
        depth_array = (depth_array - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])

        return rgb_array, depth_array


class HammerGraspRGBDEnv:
    """RGBD vision-based hammer grasping environment."""

    def __init__(self,
                 scene_path="hammer_grasp_scene.xml",
                 image_size: Tuple[int, int] = (320, 240),
                 num_cameras: int = 2,
                 action_smoothing: float = 0.3,
                 smoothness_weight: float = 1.0,
                 action_scale: float = 0.5,
                 sim_substeps: int = 10,
                 use_hand_control: bool = True,
                 stack_frames: int = 4):
        """Initialize RGBD-based environment.

        Args:
            scene_path: Path to MuJoCo XML scene
            image_size: (width, height) of rendered images
            num_cameras: Number of cameras (1 or 2 for stereo)
            action_smoothing: Action smoothing coefficient
            smoothness_weight: Smoothness penalty weight
            action_scale: Action scaling factor
            sim_substeps: Physics steps per environment step
            use_hand_control: Include hand/finger actuators
            stack_frames: Number of consecutive frames to stack for temporal info
        """
        self.scene_path = scene_path
        self.image_size = image_size
        self.num_cameras = num_cameras
        self.action_smoothing = action_smoothing
        self.smoothness_weight = smoothness_weight
        self.action_scale = action_scale
        self.sim_substeps = sim_substeps
        self.use_hand_control = use_hand_control
        self.stack_frames = stack_frames

        if not os.path.exists(self.scene_path):
            raise FileNotFoundError(f"Scene file not found: {self.scene_path}")

        # Load model
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        print(f"Loaded model: {self.model.nbody} bodies, {self.model.njnt} joints")

        # Initialize cameras
        self.cameras = []
        camera_names = ["track_left", "track_right"] if num_cameras == 2 else ["track"]

        for cam_name in camera_names:
            try:
                cam = RGBDCamera(self.model, self.data, cam_name, image_size)
                self.cameras.append(cam)
                print(f"✓ Initialized camera: {cam_name}")
            except ValueError as e:
                print(f"⚠ Warning: {e} - using synthetic camera instead")
                # Fallback: create camera anyway (will render blank)
                break

        if not self.cameras:
            print("⚠ No cameras found in scene, using dummy camera for synthetic images")

        # Setup actuators (same as proprioceptive version)
        self._setup_actuators()

        # Episode parameters
        self.max_episode_steps = 800
        self.current_step = 0

        # Frame stacking for temporal information
        self.frame_buffer = deque(maxlen=stack_frames)

        # State tracking
        self.filtered_action = None
        self.last_action = None
        self.contact_with_hammer = False
        self.grasp_contact_frames = 0

        print(f"Environment initialized with {len(self.cameras)} camera(s)")
        print(f"RGBD image size: {image_size[0]}x{image_size[1]}")
        print(f"Frame stacking: {stack_frames} frames")

    def _setup_actuators(self):
        """Setup controllable actuators (same as proprioceptive version)."""
        self.controllable_actuators = []
        self.leg_actuators = []
        self.leg_joint_ids = []

        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                actuator_lower = actuator_name.lower()

                # Lock leg actuators
                if any(leg_part in actuator_lower for leg_part in ['hip', 'knee', 'ankle', 'leg']):
                    self.leg_actuators.append(i)
                    joint_id = self.model.actuator_trnid[i, 0]
                    if joint_id >= 0 and joint_id not in self.leg_joint_ids:
                        self.leg_joint_ids.append(joint_id)
                    continue

                # Control arms
                if any(arm_part in actuator_lower for arm_part in ['shoulder', 'elbow', 'wrist', 'arm']):
                    self.controllable_actuators.append(i)
                    continue

                # Control hands if enabled
                if self.use_hand_control and any(hand_part in actuator_lower for hand_part in ['hand', 'thumb', 'index', 'middle']):
                    self.controllable_actuators.append(i)
                    continue

                # Control torso
                if any(torso_part in actuator_lower for torso_part in ['torso', 'waist', 'spine']):
                    self.controllable_actuators.append(i)

        self.n_actions = len(self.controllable_actuators)
        print(f"Controllable actuators: {self.n_actions}")

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observation.

        Returns:
            Dictionary with 'rgbd' and 'proprioceptive' observations
        """
        mujoco.mj_resetData(self.model, self.data)

        # Reset leg positions
        for joint_id in self.leg_joint_ids:
            if joint_id >= 0 and joint_id < self.model.nq:
                self.data.qpos[joint_id] = 0.0

        # Reset hammer
        hammer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hammer")
        if hammer_id >= 0:
            base_pos = np.array([0.5, 0.0, 0.75])
            noise = np.random.uniform(-0.05, 0.05, 3)
            noise[2] = 0
            self.data.xpos[hammer_id][:] = base_pos + noise

        mujoco.mj_forward(self.model, self.data)

        # Reset state
        self.current_step = 0
        self.filtered_action = np.zeros(self.n_actions)
        self.last_action = np.zeros(self.n_actions)
        self.contact_with_hammer = False
        self.grasp_contact_frames = 0
        self.frame_buffer.clear()

        # Get initial observations
        return self._get_observation()

    def _get_rgbd_observation(self) -> Dict[str, np.ndarray]:
        """Capture RGBD images from all cameras.

        Returns:
            Dictionary with 'rgb' and 'depth' keys containing stacked images
        """
        rgb_images = []
        depth_images = []

        # Render from all cameras
        for camera in self.cameras:
            try:
                rgb, depth = camera.render()
                rgb_images.append(rgb)
                depth_images.append(depth)
            except Exception as e:
                print(f"Warning: Camera rendering failed: {e}")
                # Use zeros as fallback
                rgb_images.append(np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8))
                depth_images.append(np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32))

        # Stack images
        if len(rgb_images) == 1:
            stacked_rgb = rgb_images[0]
            stacked_depth = depth_images[0]
        else:
            # Concatenate horizontally for stereo (or could stack as separate channels)
            stacked_rgb = np.hstack(rgb_images)  # (H, W*num_cameras, 3)
            stacked_depth = np.hstack(depth_images)  # (H, W*num_cameras)

        # Store frame for temporal stacking
        # Create RGBD by concatenating as 4-channel image (H, W, 4)
        if len(rgb_images) == 1:
            frame = np.concatenate([rgb_images[0], depth_images[0][:, :, np.newaxis]], axis=2)
        else:
            # For stereo, combine into (H, W*2, 4) or process differently
            frame = np.dstack([stacked_rgb, stacked_depth[:, :, np.newaxis]])

        self.frame_buffer.append(frame)

        # Pad buffer with first frame if not full
        while len(self.frame_buffer) < self.stack_frames:
            self.frame_buffer.append(self.frame_buffer[0] if len(self.frame_buffer) > 0 else np.zeros_like(frame))

        # Stack frames along channel dimension
        stacked_frames = np.concatenate(list(self.frame_buffer), axis=2)  # (H, W, C*stack_frames)

        return {
            'rgb': stacked_rgb,
            'depth': stacked_depth,
            'rgbd_stacked': stacked_frames,  # (H, W, 4*stack_frames) or similar
        }

    def _get_proprioceptive_observation(self) -> np.ndarray:
        """Get low-level proprioceptive state.

        Returns:
            Joint positions and velocities for proprioceptive feedback
        """
        # Extract arm joint angles and velocities
        arm_qpos = []
        arm_qvel = []

        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and any(part in joint_name for part in ['shoulder', 'elbow', 'wrist']):
                qpos_addr = self.model.jnt_qposadr[i]
                qvel_addr = self.model.jnt_dofadr[i]

                if qpos_addr >= 0 and qpos_addr < self.model.nq:
                    arm_qpos.append(self.data.qpos[qpos_addr])
                if qvel_addr >= 0 and qvel_addr < self.model.nv:
                    arm_qvel.append(self.data.qvel[qvel_addr])

        # Pad or truncate to 14 dimensions each
        arm_qpos = np.array(arm_qpos[:14]) if arm_qpos else np.zeros(14)
        arm_qvel = np.array(arm_qvel[:14]) if arm_qvel else np.zeros(14)

        # Combine into proprioceptive vector
        proprio = np.concatenate([arm_qpos, arm_qvel])  # 28-dimensional

        return proprio.astype(np.float32)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get complete observation including RGBD and proprioceptive feedback.

        Returns:
            Dictionary with 'rgbd', 'rgbd_stacked', and 'proprioceptive' keys
        """
        rgbd_obs = self._get_rgbd_observation()
        proprio_obs = self._get_proprioceptive_observation()

        return {
            'rgb': rgbd_obs['rgb'].astype(np.uint8),
            'depth': rgbd_obs['depth'].astype(np.float32),
            'rgbd_stacked': rgbd_obs['rgbd_stacked'].astype(np.float32),
            'proprioceptive': proprio_obs,
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute one environment step.

        Args:
            action: Control commands

        Returns:
            Tuple of (observation, reward, done, info)
        """
        action = np.clip(action, -1.0, 1.0)

        # Action smoothing
        if self.filtered_action is None:
            self.filtered_action = action.copy()
        else:
            self.filtered_action = (self.action_smoothing * action +
                                   (1 - self.action_smoothing) * self.filtered_action)

        scaled_action = self.filtered_action * self.action_scale

        # Lock legs
        for leg_act_idx in self.leg_actuators:
            joint_id = self.model.actuator_trnid[leg_act_idx, 0]
            if joint_id >= 0:
                self.data.ctrl[leg_act_idx] = 0.0

        # Apply arm/hand control
        for i, act_idx in enumerate(self.controllable_actuators):
            if i < len(scaled_action):
                self.data.ctrl[act_idx] = scaled_action[i]

        # Physics steps
        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)

        # Check contact
        hammer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hammer")
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

        # Reward
        reward = self._compute_reward(action)

        # Done check
        done = self.current_step >= self.max_episode_steps
        self.current_step += 1

        obs = self._get_observation()

        info = {
            'step': self.current_step,
            'contact_with_hammer': self.contact_with_hammer,
        }

        return obs, reward, done, info

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward (same logic as proprioceptive version)."""
        reward = 0.0

        # Get hand position (from visual tracking or proprioception)
        right_wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        if right_wrist_id >= 0:
            right_hand_pos = self.data.xpos[right_wrist_id]
        else:
            right_hand_pos = np.zeros(3)

        hammer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hammer")
        if hammer_id >= 0:
            hammer_pos = self.data.xpos[hammer_id]
        else:
            hammer_pos = np.zeros(3)

        hand_to_hammer_dist = np.linalg.norm(right_hand_pos - hammer_pos)
        distance_reward = -hand_to_hammer_dist

        contact_reward = 0.0
        if self.contact_with_hammer:
            self.grasp_contact_frames += 1
            contact_reward = 0.5
        else:
            self.grasp_contact_frames = 0

        grasp_reward = 0.0
        if self.grasp_contact_frames > 10:
            grasp_reward = 1.0

        action_diff = np.linalg.norm(action - self.last_action) if self.last_action is not None else 0.0
        smoothness_penalty = -self.smoothness_weight * action_diff * 0.01

        self.last_action = action.copy()

        reward = distance_reward * 0.5 + contact_reward * 0.3 + grasp_reward * 0.2 + smoothness_penalty

        return float(reward)

    def close(self):
        """Close environment."""
        pass

    @property
    def observation_space(self):
        """Get observation space info."""
        return {
            'rgb': (self.image_size[1], self.image_size[0], 3 * self.num_cameras),
            'depth': (self.image_size[1], self.image_size[0] * self.num_cameras),
            'rgbd_stacked': (self.image_size[1], self.image_size[0] * self.num_cameras, 4 * self.stack_frames),
            'proprioceptive': (28,),
        }

    @property
    def action_space(self):
        """Get action space size."""
        return self.n_actions


if __name__ == "__main__":
    print("Testing RGBD environment...")
    env = HammerGraspRGBDEnv(num_cameras=1, stack_frames=2)

    print("\nResetting...")
    obs = env.reset()

    print(f"Observation keys: {obs.keys()}")
    print(f"RGB shape: {obs['rgb'].shape}")
    print(f"Depth shape: {obs['depth'].shape}")
    print(f"RGBD stacked shape: {obs['rgbd_stacked'].shape}")
    print(f"Proprioceptive shape: {obs['proprioceptive'].shape}")

    print("\nRunning 50 steps...")
    for step in range(50):
        action = np.random.uniform(-1, 1, env.n_actions)
        obs, reward, done, info = env.step(action)

        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.4f}, contact={info['contact_with_hammer']}")

    env.close()
    print("\nTest complete!")
