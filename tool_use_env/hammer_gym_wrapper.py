#!/usr/bin/env python3
"""
Gymnasium wrapper for the hammer grasping environment.

Provides a standard Gymnasium interface for the MuJoCo hammer grasping
environment, making it compatible with Stable-Baselines3 and other RL libraries.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from hammer_grasp_environment import HammerGraspEnv


class HammerGraspGymWrapper(gym.Env):
    """Gymnasium wrapper for hammer grasping environment."""

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 scene_path="hammer_grasp_scene.xml",
                 action_smoothing=0.3,
                 smoothness_weight=1.0,
                 action_scale=0.5,
                 sim_substeps=10,
                 use_hand_control=True,
                 render_mode: Optional[str] = None):
        """Initialize the Gymnasium wrapper.

        Args:
            scene_path: Path to MuJoCo XML scene file
            action_smoothing: Action smoothing coefficient
            smoothness_weight: Smoothness penalty weight
            action_scale: Action scaling factor
            sim_substeps: Number of substeps per environment step
            use_hand_control: Whether to include hand control in actions
            render_mode: Render mode ('human' or None)
        """
        self.env = HammerGraspEnv(
            scene_path=scene_path,
            action_smoothing=action_smoothing,
            smoothness_weight=smoothness_weight,
            action_scale=action_scale,
            sim_substeps=sim_substeps,
            use_hand_control=use_hand_control
        )

        self.render_mode = render_mode

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.n_actions,),
            dtype=np.float32
        )

        # Observation space: 40 dimensions
        # Right hand pos (3) + left hand pos (3) + hammer pos (3) + hammer vel (3) +
        # arm qpos (14) + arm qvel (14)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(40,),
            dtype=np.float32
        )

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        obs = self.env.reset()
        info = {
            'episode': 0,
            'contact_with_hammer': False,
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step of the environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, done, info = self.env.step(action)

        # Gymnasium expects (terminated, truncated) instead of done
        terminated = done
        truncated = False

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == 'human':
            self.env.render(mode='human')
        return None

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


if __name__ == "__main__":
    # Test the wrapper
    print("Creating wrapped environment...")
    env = HammerGraspGymWrapper()

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    print("\nTesting 50 steps...")
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            print(f"Step {i}: reward={reward:.4f}")

        if terminated or truncated:
            print(f"Episode finished at step {i}")
            break

    env.close()
    print("\nTest complete!")
