#!/usr/bin/env python3
"""
Enhanced grasping environment with Grounding DINO vision detection.

This environment integrates zero-shot object detection for flexible,
natural language-based object recognition.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from pathlib import Path
import cv2

from grounding_dino_detector import GroundingDINODetector
from camera_utils import CameraProcessor


class GroundingDINOGraspEnv(gym.Env):
    """
    RL environment with Grounding DINO vision for object detection.

    Features:
    - Zero-shot object detection using natural language prompts
    - Geometry-based filtering (aspect ratio, size, position)
    - Both 2D bounding boxes and 3D positions
    - Configurable: can use physics ground truth or vision
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        detection_prompt="hammer",
        use_vision_for_rewards=False,  # If False, use physics ground truth
        use_geometry_filtering=True,   # Filter detections by geometry
        detection_threshold=0.30,
        use_wrist_camera=True,
        image_size=(84, 84),
        frame_stack=3,
        action_scale=0.02,
        max_steps=100,
        render_mode=None
    ):
        super().__init__()

        self.detection_prompt = detection_prompt
        self.use_vision_for_rewards = use_vision_for_rewards
        self.use_geometry_filtering = use_geometry_filtering
        self.detection_threshold = detection_threshold
        self.use_wrist_camera = use_wrist_camera
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.action_scale = action_scale
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Load MuJoCo model
        scene_path = Path(__file__).parent / "hammer_grasp_rgbd_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Camera processor
        self.camera_processor = CameraProcessor(self.model, width=640, height=480)
        self.wrist_camera_name = "right_wrist_camera"
        self.head_camera_name = "track_front"

        # Initialize Grounding DINO detector
        print(f"🤖 Initializing Grounding DINO detector...")
        self.detector = GroundingDINODetector()

        # Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper_close]
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # Observation space
        if self.use_wrist_camera:
            # Image-based observations with stacked frames
            obs_shape = (self.frame_stack, *self.image_size)
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                'proprioception': spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
                # proprioception: [gripper_pos(3), distance_to_target(1), detected_bbox(4), confidence(1), gripper_width(1)]
            })
            self.frame_buffer = []
        else:
            # State-based observations
            # [gripper_pos(3), gripper_quat(4), target_pos(3), distance(1), bbox_center(2), confidence(1)]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
            )

        # Internal state
        self.current_step = 0
        self.hammer_body_id = None
        self.gripper_site_id = None
        self.current_detection = None
        self.target_position = None

        # Statistics
        self.detection_success_count = 0
        self.detection_total_count = 0
        self.detection_errors = []

        print(f"\n{'='*60}")
        print(f"Grounding DINO Grasp Environment")
        print(f"{'='*60}")
        print(f"Detection prompt: '{detection_prompt}'")
        print(f"Vision for rewards: {use_vision_for_rewards}")
        print(f"Geometry filtering: {use_geometry_filtering}")
        print(f"Detection threshold: {detection_threshold}")
        print(f"Wrist camera: {'Enabled' if use_wrist_camera else 'Disabled'}")
        print(f"Observation space: {self.observation_space}")
        print(f"{'='*60}\n")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Get body/site IDs
        if self.hammer_body_id is None:
            try:
                self.hammer_body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "hammer"
                )
                self.gripper_site_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_palm_center"
                )
            except Exception as e:
                print(f"⚠️ Warning: Could not find hammer/gripper: {e}")
                # Try alternative names
                try:
                    self.gripper_site_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm"
                    )
                except:
                    self.gripper_site_id = -1

        # Let scene settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Get ground truth position (always available for comparison)
        hammer_pos_gt = self._get_hammer_position_physics()

        # Detect object with vision
        self.current_detection = self._detect_object()

        if self.current_detection is not None:
            self.detection_success_count += 1

            # Calculate detection error
            if hammer_pos_gt is not None:
                error = np.linalg.norm(self.current_detection['position_3d'] - hammer_pos_gt)
                self.detection_errors.append(error)

                # Print stats occasionally
                if self.detection_total_count % 10 == 0 and self.detection_total_count > 0:
                    success_rate = self.detection_success_count / self.detection_total_count
                    avg_error = np.mean(self.detection_errors[-10:]) if len(self.detection_errors) > 0 else 0
                    print(f"Vision stats: {success_rate:.1%} success, ±{avg_error*100:.1f}cm error")

            # Set target position
            if self.use_vision_for_rewards:
                self.target_position = self.current_detection['position_3d']
            else:
                self.target_position = hammer_pos_gt
        else:
            # Fall back to physics
            self.target_position = hammer_pos_gt
            if self.detection_total_count % 10 == 0:
                print(f"⚠️ No detection, using physics ground truth")

        self.detection_total_count += 1
        self.current_step = 0

        # Clear frame buffer
        if self.use_wrist_camera:
            self.frame_buffer = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        # Apply action (delta control)
        self._apply_action(action)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._compute_reward()

        # Check termination
        terminated = self._check_grasp_success()
        truncated = self.current_step >= self.max_steps

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _detect_object(self):
        """
        Detect object using Grounding DINO.

        Returns dict with:
        - bbox: [x1, y1, x2, y2]
        - confidence: detection confidence
        - position_3d: 3D position in world frame
        - geometry: aspect ratio, size, etc.
        """
        # Capture RGB-D from head camera
        rgb, depth = self._capture_head_camera_rgbd()

        # Run detection
        detections = self.detector.detect(
            rgb,
            text_prompt=self.detection_prompt,
            box_threshold=self.detection_threshold
        )

        if len(detections) == 0:
            return None

        # Apply geometry filtering if enabled
        if self.use_geometry_filtering:
            detections = self._filter_detections(detections, rgb.shape)

        if len(detections) == 0:
            return None

        # Get best detection
        best = detections[0]

        # Get 3D position
        K = self.camera_processor.get_camera_intrinsics(self.head_camera_name)
        pos_3d_cam = self.detector.get_3d_position(best, depth, K)

        # Convert to world frame
        pos_3d_world = self.camera_processor.camera_to_world_frame(
            pos_3d_cam.reshape(1, 3), self.data, self.head_camera_name
        )[0]

        # Calculate geometry features
        x1, y1, x2, y2 = best['bbox']
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / max(height, 1)

        return {
            'bbox': best['bbox'],
            'bbox_normalized': best['bbox_normalized'],
            'confidence': best['confidence'],
            'position_3d': pos_3d_world,
            'center_2d': best['center'],
            'aspect_ratio': aspect_ratio,
            'size_2d': (width, height)
        }

    def _filter_detections(self, detections, image_shape):
        """
        Filter detections to find the actual target object.

        For hammer detection, filters out robot arms by:
        - Aspect ratio (hammer is more horizontal, ratio > 2.0)
        - Position (hammer is in upper part of image)
        - Size (hammer is smaller than robot arms)
        """
        filtered = []

        image_height = image_shape[0]

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(height, 1)

            # Filter criteria for hammer
            # Adjust these based on your specific object!
            is_horizontal = aspect_ratio > 2.0  # Hammer is elongated
            is_upper_image = y1 < image_height / 2  # Hammer is further away
            is_reasonable_size = width < image_shape[1] * 0.3  # Not too large

            if is_horizontal and is_upper_image and is_reasonable_size:
                filtered.append(det)

        return filtered

    def _get_observation(self):
        if self.use_wrist_camera:
            # Capture wrist camera image
            rgb = self._capture_wrist_camera()

            # Convert to grayscale and resize
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, self.image_size)

            # Add to frame buffer
            self.frame_buffer.append(gray)
            if len(self.frame_buffer) > self.frame_stack:
                self.frame_buffer.pop(0)

            # Pad if needed
            while len(self.frame_buffer) < self.frame_stack:
                self.frame_buffer.append(gray)

            # Stack frames
            image_obs = np.stack(self.frame_buffer, axis=0)

            # Get proprioception
            proprio = self._get_proprioception()

            return {
                'image': image_obs.astype(np.uint8),
                'proprioception': proprio.astype(np.float32)
            }
        else:
            # State-based observation
            gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
            gripper_quat = np.array([0, 0, 0, 1])  # Simplified

            target_pos = self.target_position
            distance = np.linalg.norm(gripper_pos - target_pos)

            # Detection info
            if self.current_detection is not None:
                bbox_center = np.array(self.current_detection['center_2d'], dtype=np.float32)
                confidence = self.current_detection['confidence']
            else:
                bbox_center = np.array([0, 0], dtype=np.float32)
                confidence = 0.0

            obs = np.concatenate([
                gripper_pos,      # 3
                gripper_quat,     # 4
                target_pos,       # 3
                [distance],       # 1
                bbox_center,      # 2
                [confidence]      # 1
            ])

            return obs.astype(np.float32)

    def _get_proprioception(self):
        """Get robot proprioceptive state."""
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        distance = np.linalg.norm(gripper_pos - self.target_position)

        # Detection features
        if self.current_detection is not None:
            bbox = np.array(self.current_detection['bbox_normalized'], dtype=np.float32)
            confidence = self.current_detection['confidence']
        else:
            bbox = np.zeros(4, dtype=np.float32)
            confidence = 0.0

        gripper_width = 0.04  # Placeholder

        return np.concatenate([
            gripper_pos,           # 3
            [distance],            # 1
            bbox,                  # 4
            [confidence],          # 1
            [gripper_width]        # 1
        ])

    def _capture_wrist_camera(self):
        """Capture RGB image from wrist-mounted camera."""
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data, camera=self.wrist_camera_name)
        rgb = renderer.render()
        return rgb

    def _capture_head_camera_rgbd(self):
        """Capture RGB-D from head/track camera."""
        renderer = mujoco.Renderer(self.model, height=480, width=640)

        # RGB
        renderer.update_scene(self.data, camera=self.head_camera_name)
        rgb = renderer.render()

        # Depth
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=self.head_camera_name)
        depth_raw = renderer.render()
        renderer.disable_depth_rendering()

        # Convert depth to meters
        # MuJoCo depth is in range [near, far], need to denormalize
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.head_camera_name)
        znear = self.model.vis.map.znear * self.model.stat.extent
        zfar = self.model.vis.map.zfar * self.model.stat.extent

        # depth_raw is in [0, 1], convert to meters
        depth = znear / (1.0 - depth_raw * (1.0 - znear / zfar))

        return rgb, depth

    def _get_hammer_position_physics(self):
        """Get ground truth hammer position using physics."""
        if self.hammer_body_id >= 0:
            return self.data.xpos[self.hammer_body_id].copy()
        else:
            return np.array([0.9, 0.0, 0.7])  # Default

    def _apply_action(self, action):
        """Apply delta control action to gripper."""
        # Get current gripper position
        current_pos = self.data.site_xpos[self.gripper_site_id].copy()

        # Apply delta position (first 3 actions)
        delta_pos = action[:3] * self.action_scale
        new_pos = current_pos + delta_pos

        # Clamp to workspace
        new_pos = np.clip(new_pos, [0.5, -0.5, 0.6], [1.2, 0.5, 1.2])

        # Apply to robot (simplified - you'll need actual IK/control)
        # This is a placeholder - replace with your actual robot control
        # For now, directly set end-effector position (teleportation)
        # In practice, you'd use IK or joint velocity control

        # Gripper control (last action)
        gripper_cmd = action[-1]  # -1 = open, +1 = close

        # TODO: Implement actual robot control here
        pass

    def _compute_reward(self):
        """Compute reward based on distance to target and grasp success."""
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        target_pos = self.target_position

        # Distance reward (sparse)
        distance = np.linalg.norm(gripper_pos - target_pos)
        distance_reward = -distance  # Negative distance

        # Bonus for being close
        if distance < 0.05:  # Within 5cm
            distance_reward += 1.0

        # Bonus for successful grasp
        if self._check_grasp_success():
            distance_reward += 10.0

        # Small penalty for each step (encourage efficiency)
        time_penalty = -0.01

        return distance_reward + time_penalty

    def _check_grasp_success(self):
        """Check if grasp was successful (hammer lifted)."""
        if self.hammer_body_id < 0:
            return False

        hammer_pos = self.data.xpos[self.hammer_body_id]
        hammer_height = hammer_pos[2]

        # Success if hammer is lifted above threshold
        success_height = 0.8  # 80cm above ground
        return hammer_height > success_height

    def _get_info(self):
        """Get additional info for logging."""
        info = {
            'step': self.current_step,
            'detection_success': self.current_detection is not None,
        }

        if self.current_detection is not None:
            info['detection_confidence'] = self.current_detection['confidence']
            info['detection_aspect_ratio'] = self.current_detection['aspect_ratio']

        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        info['distance_to_target'] = np.linalg.norm(gripper_pos - self.target_position)

        return info

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data, camera=self.head_camera_name)
            return renderer.render()
        return None

    def close(self):
        pass


# Test function
if __name__ == "__main__":
    print("Testing Grounding DINO Grasp Environment...")

    env = GroundingDINOGraspEnv(
        detection_prompt="hammer",
        use_vision_for_rewards=False,  # Start with physics ground truth
        use_geometry_filtering=True,
        detection_threshold=0.30,
        use_wrist_camera=False  # Use state observations for faster testing
    )

    print("\nRunning test episode...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Distance: {info['distance_to_target']:.3f}m")
        if info['detection_success']:
            print(f"  Detection: {info['detection_confidence']:.1%} confidence")

        if terminated or truncated:
            break

    print("\n✓ Test complete!")
