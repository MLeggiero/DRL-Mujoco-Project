#!/usr/bin/env python3
"""
Vision-Guided Grasping Environment with Multi-Object Detection.

This environment uses the corrected multi-object detector to:
1. Detect tools in the scene
2. Provide target positions for grasping
3. Train RL policies for tool manipulation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from pathlib import Path
import cv2
from typing import Optional, Dict, Tuple

from multi_object_detector import MultiObjectDetector
from camera_utils import CameraProcessor


class VisionGuidedGraspEnv(gym.Env):
    """
    RL environment for vision-guided grasping.

    Features:
    - Multi-object detection (tools + hands)
    - Configurable curriculum learning
    - Multiple reward modes
    - Hand-eye coordination tracking
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        # Task settings
        target_object: str = "hammer",
        task_mode: str = "grasp",  # "grasp", "pick_place", "handover"

        # Vision settings
        use_vision_detection: bool = True,
        vision_update_freq: int = 1,  # Detect every N episodes
        track_hands: bool = True,

        # Reward settings
        reward_mode: str = "hybrid",  # "physics", "vision", "hybrid"
        reward_shaping: str = "dense",  # "dense", "sparse"

        # Environment settings
        action_scale: float = 0.02,
        max_steps: int = 100,
        observation_mode: str = "state",  # "state", "image", "both"
        image_size: Tuple[int, int] = (84, 84),

        # Difficulty/curriculum
        detection_threshold: float = 0.30,
        add_noise: bool = False,
        noise_std: float = 0.01,

        render_mode: Optional[str] = None
    ):
        super().__init__()

        # Store parameters
        self.target_object = target_object
        self.task_mode = task_mode
        self.use_vision_detection = use_vision_detection
        self.vision_update_freq = vision_update_freq
        self.track_hands = track_hands
        self.reward_mode = reward_mode
        self.reward_shaping = reward_shaping
        self.action_scale = action_scale
        self.max_steps = max_steps
        self.observation_mode = observation_mode
        self.image_size = image_size
        self.detection_threshold = detection_threshold
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.render_mode = render_mode

        # Load MuJoCo model
        scene_path = Path(__file__).parent / "hammer_grasp_rgbd_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Camera setup
        self.camera_processor = CameraProcessor(self.model, width=640, height=480)
        self.head_camera_name = "track_front"
        self.wrist_camera_name = "right_wrist_camera"

        # Initialize vision detector
        if self.use_vision_detection:
            print(f"🤖 Initializing multi-object detector for {target_object}...")
            self.detector = MultiObjectDetector(verbose=False)
        else:
            self.detector = None

        # Get body/site IDs
        self.hammer_body_id = None
        self.gripper_site_id = None
        self._initialize_ids()

        # Define spaces
        self._setup_spaces()

        # Episode state
        self.current_step = 0
        self.target_position = None
        self.target_detected = False
        self.gripper_position = None
        self.hand_detections = []
        self.tool_detections = []
        self.episode_count = 0
        self.last_detection_episode = 0

        # Statistics
        self.detection_success_rate = []
        self.grasp_success_rate = []

        print(f"\n{'='*60}")
        print(f"Vision-Guided Grasping Environment")
        print(f"{'='*60}")
        print(f"Target object: {target_object}")
        print(f"Task mode: {task_mode}")
        print(f"Vision detection: {use_vision_detection}")
        print(f"Reward mode: {reward_mode} ({reward_shaping})")
        print(f"Observation: {observation_mode}")
        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space}")
        print(f"{'='*60}\n")

    def _initialize_ids(self):
        """Get MuJoCo body and site IDs."""
        try:
            self.hammer_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "hammer"
            )
        except:
            self.hammer_body_id = -1

        try:
            self.gripper_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_palm_center"
            )
        except:
            try:
                self.gripper_site_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm"
                )
            except:
                self.gripper_site_id = -1

    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # Observation space
        if self.observation_mode == "state":
            # State-based: gripper + target + detection info
            obs_dim = (
                3 +  # gripper position
                4 +  # gripper quaternion
                3 +  # target position
                1 +  # distance to target
                1 +  # detection confidence
                1 +  # gripper width
                3    # hand position (if tracked)
            )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

        elif self.observation_mode == "image":
            # Image-based: wrist camera
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(3, *self.image_size),  # RGB
                dtype=np.uint8
            )

        else:  # "both"
            # Both state and image
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0, high=255,
                    shape=(3, *self.image_size),
                    dtype=np.uint8
                ),
                'state': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(16,),
                    dtype=np.float32
                )
            })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Let scene settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Get ground truth position
        hammer_pos_gt = self._get_hammer_position_physics()

        # Run vision detection (if enabled and due)
        if self.use_vision_detection and (
            self.episode_count - self.last_detection_episode >= self.vision_update_freq
        ):
            self._detect_scene()
            self.last_detection_episode = self.episode_count

        # Set target position based on reward mode
        if self.reward_mode == "vision" and self.target_detected:
            # Use vision detection
            self.target_position = self.tool_detections[0]['position_3d']
        elif self.reward_mode == "hybrid" and self.target_detected:
            # Mix vision and physics (reduces noise)
            vision_pos = self.tool_detections[0]['position_3d']
            mix_factor = 0.7  # 70% vision, 30% physics
            self.target_position = (
                mix_factor * vision_pos + (1 - mix_factor) * hammer_pos_gt
            )
        else:
            # Use physics ground truth
            self.target_position = hammer_pos_gt

        # Add noise if enabled (for robustness)
        if self.add_noise:
            noise = np.random.normal(0, self.noise_std, size=3)
            self.target_position = self.target_position + noise

        # Get initial gripper position
        self.gripper_position = self.data.site_xpos[self.gripper_site_id].copy()

        self.current_step = 0
        self.episode_count += 1

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray):
        """Execute action and return next state."""
        # Apply action
        self._apply_action(action)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Update gripper position
        self.gripper_position = self.data.site_xpos[self.gripper_site_id].copy()

        self.current_step += 1

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._compute_reward()

        # Check termination
        terminated = self._check_success()
        truncated = self.current_step >= self.max_steps

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _detect_scene(self):
        """Run vision detection for tools and hands."""
        if self.detector is None:
            return

        # Capture RGB-D
        rgb, depth = self._capture_head_camera_rgbd()

        # Detect scene
        scene = self.detector.detect_scene(
            rgb,
            include_tools=True,
            include_hands=self.track_hands,
            include_objects=False,
            tool_threshold=self.detection_threshold,
            hand_threshold=0.25
        )

        self.tool_detections = []
        self.hand_detections = []

        # Process tool detections
        if len(scene['tools']) > 0:
            self.target_detected = True
            K = self.camera_processor.get_camera_intrinsics(self.head_camera_name)

            for det in scene['tools']:
                # Get 3D position
                pos_3d_cam = self.detector.get_3d_position(det, depth, K)
                pos_3d_world = self.camera_processor.camera_to_world_frame(
                    pos_3d_cam.reshape(1, 3), self.data, self.head_camera_name
                )[0]

                det['position_3d'] = pos_3d_world
                self.tool_detections.append(det)

            # Track detection stats
            self.detection_success_rate.append(1.0)
        else:
            self.target_detected = False
            self.detection_success_rate.append(0.0)

        # Process hand detections
        if self.track_hands and len(scene['hands']) > 0:
            K = self.camera_processor.get_camera_intrinsics(self.head_camera_name)

            for det in scene['hands']:
                pos_3d_cam = self.detector.get_3d_position(det, depth, K)
                pos_3d_world = self.camera_processor.camera_to_world_frame(
                    pos_3d_cam.reshape(1, 3), self.data, self.head_camera_name
                )[0]

                det['position_3d'] = pos_3d_world
                self.hand_detections.append(det)

    def _get_observation(self):
        """Get observation based on observation mode."""
        if self.observation_mode == "state":
            return self._get_state_observation()
        elif self.observation_mode == "image":
            return self._get_image_observation()
        else:  # "both"
            return {
                'image': self._get_image_observation(),
                'state': self._get_state_observation()
            }

    def _get_state_observation(self):
        """Get state-based observation."""
        # Gripper state
        gripper_pos = self.gripper_position
        gripper_quat = np.array([0, 0, 0, 1])  # Simplified
        gripper_width = 0.04  # Placeholder

        # Target state
        target_pos = self.target_position
        distance = np.linalg.norm(gripper_pos - target_pos)

        # Detection confidence
        if len(self.tool_detections) > 0:
            confidence = self.tool_detections[0]['confidence']
        else:
            confidence = 0.0

        # Hand position (if tracked)
        if self.track_hands and len(self.hand_detections) > 0:
            hand_pos = self.hand_detections[0]['position_3d']
        else:
            hand_pos = gripper_pos  # Use gripper as fallback

        obs = np.concatenate([
            gripper_pos,      # 3
            gripper_quat,     # 4
            target_pos,       # 3
            [distance],       # 1
            [confidence],     # 1
            [gripper_width],  # 1
            hand_pos          # 3
        ])

        return obs.astype(np.float32)

    def _get_image_observation(self):
        """Get image-based observation from wrist camera."""
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data, camera=self.wrist_camera_name)
        rgb = renderer.render()

        # Resize and transpose to CHW format
        rgb_resized = cv2.resize(rgb, self.image_size)
        rgb_chw = np.transpose(rgb_resized, (2, 0, 1))  # HWC -> CHW

        return rgb_chw.astype(np.uint8)

    def _compute_reward(self):
        """Compute reward based on reward mode and shaping."""
        distance = np.linalg.norm(self.gripper_position - self.target_position)

        if self.reward_shaping == "sparse":
            # Sparse reward: only on success
            if self._check_success():
                return 10.0
            else:
                return -0.01  # Small time penalty

        else:  # "dense"
            # Dense reward: shaped by distance
            reward = -distance  # Negative distance

            # Bonus for being close
            if distance < 0.10:  # Within 10cm
                reward += 0.5
            if distance < 0.05:  # Within 5cm
                reward += 1.0

            # Bonus for successful grasp
            if self._check_success():
                reward += 10.0

            # Small time penalty
            reward -= 0.01

            # Detection confidence bonus (if using vision)
            if self.use_vision_detection and len(self.tool_detections) > 0:
                confidence = self.tool_detections[0]['confidence']
                if confidence > 0.5:
                    reward += 0.1  # Bonus for high confidence detection

            return reward

    def _check_success(self):
        """Check if grasp was successful."""
        if self.hammer_body_id < 0:
            return False

        hammer_pos = self.data.xpos[self.hammer_body_id]
        hammer_height = hammer_pos[2]

        # Success if hammer is lifted
        success_height = 0.8  # 80cm above ground
        return hammer_height > success_height

    def _apply_action(self, action: np.ndarray):
        """Apply action to robot (placeholder - implement actual control)."""
        # TODO: Implement actual robot control
        # For now, this is a placeholder
        # In practice, use IK solver or joint velocity control
        pass

    def _capture_head_camera_rgbd(self):
        """Capture RGB-D from head camera."""
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
        znear = self.model.vis.map.znear * self.model.stat.extent
        zfar = self.model.vis.map.zfar * self.model.stat.extent
        depth = znear / (1.0 - depth_raw * (1.0 - znear / zfar))

        return rgb, depth

    def _get_hammer_position_physics(self):
        """Get ground truth hammer position."""
        if self.hammer_body_id >= 0:
            return self.data.xpos[self.hammer_body_id].copy()
        else:
            return np.array([0.9, 0.0, 0.7])

    def _get_info(self):
        """Get additional info for logging."""
        info = {
            'current_step': self.current_step,
            'episode_num': self.episode_count,
            'target_detected': self.target_detected,
            'distance_to_target': float(np.linalg.norm(self.gripper_position - self.target_position)),
        }

        if len(self.tool_detections) > 0:
            info['detection_confidence'] = float(self.tool_detections[0]['confidence'])
            info['detection_aspect_ratio'] = float(self.tool_detections[0]['geometry']['aspect_ratio'])

        if len(self.hand_detections) > 0:
            info['hands_detected'] = len(self.hand_detections)

        # Statistics
        if len(self.detection_success_rate) > 0:
            info['detection_success_rate'] = float(np.mean(self.detection_success_rate[-100:]))

        return info

    def render(self):
        """Render environment."""
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data, camera=self.head_camera_name)
            return renderer.render()
        return None

    def close(self):
        """Clean up resources."""
        pass


# Quick test
if __name__ == "__main__":
    print("Testing Vision-Guided Grasping Environment...")

    env = VisionGuidedGraspEnv(
        target_object="hammer",
        use_vision_detection=True,
        reward_mode="hybrid",
        observation_mode="state"
    )

    print("\nRunning test episode...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Distance: {info['distance']:.3f}m")
        print(f"  Target detected: {info['target_detected']}")

        if terminated or truncated:
            break

    print("\n✓ Test complete!")
