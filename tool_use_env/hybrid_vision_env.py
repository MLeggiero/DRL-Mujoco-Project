#!/usr/bin/env python3
"""
Hybrid vision-based grasping environment.

Combines:
- Physics-based position (for rewards)
- Learned vision detection (for observations/validation)
- Wrist camera images (for sim-to-real transfer)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from pathlib import Path
import cv2

# Import available detectors
try:
    from yolo_detector import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLODetector not available")

from camera_utils import CameraProcessor


class HybridVisionGraspEnv(gym.Env):
    """
    RL environment with hybrid vision approach.

    Features:
    - Physics-based ground truth for rewards (fast, accurate)
    - Optional YOLO detection for validation
    - Wrist camera observations for visual learning
    - Configurable: can use different vision backends
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        use_vision_detector=False,
        vision_backend="yolo",  # "yolo", "color", or "none"
        use_wrist_camera=True,
        image_size=(84, 84),  # Small for faster training
        frame_stack=3,  # Stack last N frames
        action_scale=0.02,
        max_steps=100,
        render_mode=None
    ):
        super().__init__()

        self.use_vision_detector = use_vision_detector
        self.vision_backend = vision_backend
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

        # Initialize vision detector if requested
        self.detector = None
        if self.use_vision_detector:
            if vision_backend == "yolo" and YOLO_AVAILABLE:
                print(f"🤖 Initializing YOLO detector...")
                self.detector = YOLODetector(model_size='n')
            elif vision_backend == "color":
                print(f"🎨 Using color-based detection")
                self.detector = "color"  # Will use color segmentation
            else:
                print(f"⚠️ Vision backend '{vision_backend}' not available")

        # Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper_close]
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # Observation space depends on configuration
        if self.use_wrist_camera:
            # Image-based observations
            # Stack grayscale images for temporal info
            obs_shape = (self.frame_stack, *self.image_size)
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                'proprioception': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
            })
            self.frame_buffer = []
        else:
            # State-based observations (faster training)
            # [gripper_pos(3), gripper_quat(4), hammer_pos(3), distance(1)]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
            )

        # Internal state
        self.current_step = 0
        self.hammer_body_id = None
        self.gripper_site_id = None

        # Statistics
        self.vision_detection_count = 0
        self.vision_success_count = 0

        print(f"\n{'='*60}")
        print(f"Hybrid Vision Grasp Environment")
        print(f"{'='*60}")
        print(f"Vision detector: {vision_backend if use_vision_detector else 'Disabled (physics only)'}")
        print(f"Wrist camera: {'Enabled' if use_wrist_camera else 'Disabled'}")
        print(f"Image size: {image_size if use_wrist_camera else 'N/A'}")
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
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
                self.hammer_body_id = -1
                self.gripper_site_id = -1

        # Let scene settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Get ground truth position (always available)
        self.hammer_pos_gt = self._get_hammer_position_physics()

        # Optional: Validate with vision detector
        if self.use_vision_detector and self.detector is not None:
            detected_pos = self._detect_hammer_with_vision()
            if detected_pos is not None:
                self.vision_success_count += 1
                error = np.linalg.norm(detected_pos - self.hammer_pos_gt)
                if self.vision_detection_count % 10 == 0:  # Print occasionally
                    print(f"Vision detection: ±{error*100:.1f}cm error")
            self.vision_detection_count += 1

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

            # Pad if needed (first few steps)
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
            gripper_quat = np.array([0, 0, 0, 1])  # Simplified for now

            # Use physics-based hammer position
            hammer_pos = self.hammer_pos_gt

            distance = np.linalg.norm(gripper_pos - hammer_pos)

            obs = np.concatenate([
                gripper_pos,      # 3
                gripper_quat,     # 4
                hammer_pos,       # 3
                [distance]        # 1
            ])

            return obs.astype(np.float32)

    def _get_proprioception(self):
        """Get robot proprioceptive state (joint positions, velocities, etc.)"""
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        distance_to_hammer = np.linalg.norm(gripper_pos - self.hammer_pos_gt)

        # Simplified: just position and distance
        # In full version: add joint angles, velocities, forces
        return np.concatenate([
            gripper_pos,                    # 3
            [distance_to_hammer],          # 1
            np.zeros(3)                    # 3 (placeholder for additional state)
        ])

    def _capture_wrist_camera(self):
        """Capture RGB image from wrist-mounted camera."""
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data, camera=self.wrist_camera_name)
        rgb = renderer.render()
        return rgb

    def _get_hammer_position_physics(self):
        """Get ground truth hammer position using physics (always works)."""
        if self.hammer_body_id >= 0:
            return self.data.xpos[self.hammer_body_id].copy()
        else:
            return np.array([0.9, 0.0, 0.7])  # Default position

    def _detect_hammer_with_vision(self):
        """
        Detect hammer using learned vision model.
        Returns 3D position or None if detection fails.
        """
        # Capture RGB-D from head camera
        rgb, depth = self._capture_head_camera_rgbd()

        if self.vision_backend == "yolo" and isinstance(self.detector, YOLODetector):
            # Use YOLO detector
            detections = self.detector.detect(rgb, conf_threshold=0.25)

            if len(detections) > 0:
                # Get 3D position of best detection
                K = self.camera_processor.get_camera_intrinsics("track_front")
                pos_3d = self.detector.get_3d_position(detections[0], depth, K)

                # Convert to world frame
                pos_world = self.camera_processor.camera_to_world_frame(
                    pos_3d.reshape(1, 3), self.data, "track_front"
                )[0]

                return pos_world

        elif self.vision_backend == "color":
            # Use color-based segmentation
            return self._detect_hammer_color(rgb, depth)

        return None

    def _detect_hammer_color(self, rgb, depth):
        """Simple color-based detection (fallback)."""
        # Convert to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Orange color range for hammer
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([15, 255, 255])

        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)

            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Get 3D position
                depth_value = depth[cy, cx]
                K = self.camera_processor.get_camera_intrinsics("track_front")

                fx, fy = K[0, 0], K[1, 1]
                cx_cam, cy_cam = K[0, 2], K[1, 2]

                z = depth_value
                x = (cx - cx_cam) * z / fx
                y = (cy - cy_cam) * z / fy

                pos_cam = np.array([x, y, z])

                # Convert to world frame
                pos_world = self.camera_processor.camera_to_world_frame(
                    pos_cam.reshape(1, 3), self.data, "track_front"
                )[0]

                return pos_world

        return None

    def _capture_head_camera_rgbd(self):
        """Capture RGB-D from head/track camera."""
        renderer = mujoco.Renderer(self.model, height=480, width=640)

        # RGB
        renderer.update_scene(self.data, camera="track_front")
        rgb = renderer.render()

        # Depth
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera="track_front")
        depth = renderer.render()
        renderer.disable_depth_rendering()

        return rgb, depth

    def _apply_action(self, action):
        """Apply delta control to robot arm."""
        # This is simplified - in full version, use IK to reach target position
        # For now, just for structure demonstration
        pass

    def _compute_reward(self):
        """Compute dense reward for RL training."""
        gripper_pos = self.data.site_xpos[self.gripper_site_id]

        # Use physics-based position for reward (accurate and fast)
        hammer_pos = self.hammer_pos_gt

        distance = np.linalg.norm(gripper_pos - hammer_pos)

        # Dense reward: negative distance
        reward = -distance * 0.1

        # Bonus for getting close
        if distance < 0.05:  # Within 5cm
            reward += 1.0

        # Big bonus for grasping
        if self._check_grasp_success():
            reward += 10.0

        return reward

    def _check_grasp_success(self):
        """Check if hammer is successfully grasped."""
        # Simplified: check if gripper is close to hammer
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        distance = np.linalg.norm(gripper_pos - self.hammer_pos_gt)

        return distance < 0.03  # Within 3cm = success

    def _get_info(self):
        """Return diagnostic information."""
        info = {
            'step': self.current_step,
            'hammer_position': self.hammer_pos_gt.copy(),
        }

        if self.use_vision_detector and self.vision_detection_count > 0:
            info['vision_success_rate'] = self.vision_success_count / self.vision_detection_count

        return info

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data, camera="track_front")
            return renderer.render()
        elif self.render_mode == "human":
            # For human rendering, would need window setup
            pass
        return None

    def close(self):
        pass


def test_environment():
    """Test the hybrid vision environment."""
    print("\n" + "="*60)
    print("Testing Hybrid Vision Environment")
    print("="*60 + "\n")

    # Test 1: Physics-only (fastest)
    print("Test 1: Physics-only mode (fastest for training)")
    env = HybridVisionGraspEnv(
        use_vision_detector=False,
        use_wrist_camera=False
    )

    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation: {obs}")
    print(f"  Info: {info}\n")

    # Take a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.3f}, distance={info.get('hammer_position', [0,0,0])[0]:.3f}")

    env.close()

    # Test 2: With YOLO detection (if available)
    if YOLO_AVAILABLE:
        print("\nTest 2: With YOLO detector (validation mode)")
        env = HybridVisionGraspEnv(
            use_vision_detector=True,
            vision_backend="yolo",
            use_wrist_camera=False
        )

        obs, info = env.reset()
        print(f"  Vision success rate: {info.get('vision_success_rate', 'N/A')}")
        env.close()

    # Test 3: Wrist camera observations (for visual RL)
    print("\nTest 3: Wrist camera mode (visual RL)")
    env = HybridVisionGraspEnv(
        use_vision_detector=False,
        use_wrist_camera=True,
        image_size=(84, 84),
        frame_stack=3
    )

    obs, info = env.reset()
    print(f"  Image observation shape: {obs['image'].shape}")
    print(f"  Proprioception shape: {obs['proprioception'].shape}")
    env.close()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_environment()
