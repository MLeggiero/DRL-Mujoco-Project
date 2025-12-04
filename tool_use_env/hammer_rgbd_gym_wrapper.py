#!/usr/bin/env python3
"""
Gymnasium wrapper for RGBD-based hammer grasping environment.

Provides standard Gymnasium interface for vision-based RL training with
Stable-Baselines3 and other compatible libraries.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from hammer_grasp_rgbd_environment import HammerGraspRGBDEnv


class HammerRGBDGymWrapper(gym.Env):
    """Gymnasium wrapper for RGBD-based hammer grasping."""

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 scene_path="hammer_grasp_rgbd_scene.xml",
                 image_size: Tuple[int, int] = (320, 240),
                 num_cameras: int = 1,
                 action_smoothing: float = 0.3,
                 smoothness_weight: float = 1.0,
                 action_scale: float = 0.5,
                 sim_substeps: int = 10,
                 use_hand_control: bool = True,
                 stack_frames: int = 4,
                 observation_mode: str = 'rgbd_stacked',
                 render_mode: Optional[str] = None):
        """Initialize RGBD Gymnasium wrapper.

        Args:
            scene_path: Path to MuJoCo scene
            image_size: (width, height) of images
            num_cameras: Number of cameras (1 or 2)
            action_smoothing: Action smoothing coefficient
            smoothness_weight: Smoothness penalty weight
            action_scale: Action scaling factor
            sim_substeps: Physics steps per environment step
            use_hand_control: Include hand actuators
            stack_frames: Number of frames to stack
            observation_mode: 'rgbd_stacked', 'rgb', 'depth', or 'rgbd_raw'
            render_mode: Render mode ('human' or None)
        """
        self.env = HammerGraspRGBDEnv(
            scene_path=scene_path,
            image_size=image_size,
            num_cameras=num_cameras,
            action_smoothing=action_smoothing,
            smoothness_weight=smoothness_weight,
            action_scale=action_scale,
            sim_substeps=sim_substeps,
            use_hand_control=use_hand_control,
            stack_frames=stack_frames
        )

        self.observation_mode = observation_mode
        self.render_mode = render_mode

        # Action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.n_actions,),
            dtype=np.float32
        )

        # Observation space depends on mode
        if observation_mode == 'rgbd_stacked':
            # Stacked RGBD frames: (H, W, C*stack_frames)
            h, w = image_size[1], image_size[0] * num_cameras
            c = 4 * stack_frames  # RGBD (4 channels) stacked
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(h, w, c),
                dtype=np.float32
            )
        elif observation_mode == 'rgb':
            # RGB only
            h, w = image_size[1], image_size[0] * num_cameras
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(h, w, 3),
                dtype=np.uint8
            )
        elif observation_mode == 'depth':
            # Depth only
            h, w = image_size[1], image_size[0] * num_cameras
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(h, w),
                dtype=np.float32
            )
        elif observation_mode == 'rgbd_raw':
            # Separate RGB and depth
            h, w = image_size[1], image_size[0] * num_cameras
            self.observation_space = spaces.Dict({
                'rgb': spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                'depth': spaces.Box(low=0, high=1, shape=(h, w), dtype=np.float32),
                'proprioceptive': spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
            })
        else:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        obs_dict = self.env.reset()
        obs = self._process_observation(obs_dict)
        info = {'episode': 0}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs_dict, reward, done, info = self.env.step(action)

        obs = self._process_observation(obs_dict)
        terminated = done
        truncated = False

        return obs, float(reward), terminated, truncated, info

    def _process_observation(self, obs_dict: Dict[str, np.ndarray]) -> Any:
        """Process observation based on observation mode.

        Args:
            obs_dict: Dictionary with 'rgb', 'depth', 'rgbd_stacked', 'proprioceptive'

        Returns:
            Processed observation
        """
        if self.observation_mode == 'rgbd_stacked':
            # Normalize RGBD to [0, 1] range and return as float32
            rgbd = obs_dict['rgbd_stacked'].astype(np.float32)
            # Normalize: RGB channels are [0, 255], depth is [0, 1]
            # For simplicity, keep as is (or normalize if needed for NN)
            return rgbd

        elif self.observation_mode == 'rgb':
            return obs_dict['rgb'].astype(np.uint8)

        elif self.observation_mode == 'depth':
            return obs_dict['depth'].astype(np.float32)

        elif self.observation_mode == 'rgbd_raw':
            return {
                'rgb': obs_dict['rgb'].astype(np.uint8),
                'depth': obs_dict['depth'].astype(np.float32),
                'proprioceptive': obs_dict['proprioceptive'].astype(np.float32)
            }

        else:
            raise ValueError(f"Unknown observation_mode: {self.observation_mode}")

    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        if self.render_mode == 'human':
            # Could launch viewer here
            pass
        return None

    def close(self) -> None:
        """Close environment."""
        self.env.close()


if __name__ == "__main__":
    print("Testing RGBD Gymnasium wrapper...")

    # Test different observation modes
    for obs_mode in ['rgbd_stacked', 'rgb', 'depth']:
        print(f"\n=== Testing observation_mode: {obs_mode} ===")
        env = HammerRGBDGymWrapper(observation_mode=obs_mode, num_cameras=1)

        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")

        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else 'dict'}")

        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if step == 0:
                print(f"Step observation shape: {obs.shape if hasattr(obs, 'shape') else 'dict'}")

        env.close()

    print("\nTest complete!")
