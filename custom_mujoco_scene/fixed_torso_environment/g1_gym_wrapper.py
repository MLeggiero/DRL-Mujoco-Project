#!/usr/bin/env python3
"""
Gymnasium wrapper for G1 Reaching Environment
Makes the environment compatible with Stable-Baselines3
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from g1_rl_environment import G1ReachTouchEnv


class G1ReachingGymEnv(gym.Env):
    """
    Gymnasium wrapper for G1 Reaching environment
    Compatible with Stable-Baselines3
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, scene_path="../unitree_g1/g1_table_box_scene.xml", render_mode=None):
        """Initialize the Gym environment"""
        super().__init__()

        self.render_mode = render_mode

        # Create the underlying environment
        self.env = G1ReachTouchEnv(scene_path=scene_path)

        # Define action space (10 actuators: 7 arm + 3 torso)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.n_actions,),
            dtype=np.float32
        )

        # Define observation space
        # Get a sample observation to determine dimensions
        sample_obs = self.env.reset()
        obs_dict = self._flatten_observation(sample_obs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(obs_dict),),
            dtype=np.float32
        )

        print(f"Gym Environment initialized:")
        print(f"  Action space: {self.action_space}")
        print(f"  Observation space: {self.observation_space}")

    def _flatten_observation(self, obs_dict):
        """Convert observation dict to flat numpy array"""
        components = []

        # Joint positions and velocities
        if 'robot_qpos' in obs_dict:
            components.append(obs_dict['robot_qpos'])
        if 'robot_qvel' in obs_dict:
            components.append(obs_dict['robot_qvel'])

        # End effector position
        if 'end_effector_pos' in obs_dict:
            components.append(obs_dict['end_effector_pos'])

        # Target position
        if 'target_position' in obs_dict:
            components.append(obs_dict['target_position'])

        # Hand to target vector
        if 'hand_to_target' in obs_dict:
            components.append(obs_dict['hand_to_target'])

        # Distance (scalar)
        if 'distance_to_target' in obs_dict:
            components.append(np.array([obs_dict['distance_to_target']]))

        # Concatenate all components
        flat_obs = np.concatenate([np.atleast_1d(c).flatten() for c in components])

        return flat_obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset the underlying environment
        obs_dict = self.env.reset()

        # Convert to flat observation
        observation = self._flatten_observation(obs_dict)

        # Gymnasium requires returning (observation, info)
        info = {}

        return observation, info

    def step(self, action):
        """Execute one step in the environment"""
        # Ensure action is numpy array
        action = np.array(action, dtype=np.float32)

        # Step the underlying environment
        obs_dict, reward, done, info = self.env.step(action)

        # Convert observation to flat array
        observation = self._flatten_observation(obs_dict)

        # Gymnasium uses (terminated, truncated) instead of just done
        terminated = info.get('success', False)
        truncated = done and not terminated  # Episode ended due to time limit

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            self.env.render()
        elif self.render_mode == "rgb_array":
            # Return RGB array if needed
            return None
        return None

    def close(self):
        """Clean up resources"""
        if hasattr(self.env, 'close'):
            self.env.close()


# Register the environment with Gymnasium
def register_g1_env():
    """Register G1 environment with Gymnasium"""
    try:
        gym.register(
            id='G1Reaching-v0',
            entry_point='g1_gym_wrapper:G1ReachingGymEnv',
            max_episode_steps=1000,
        )
        print("Successfully registered G1Reaching-v0 environment")
    except:
        print("G1Reaching-v0 already registered")


if __name__ == "__main__":
    # Test the wrapper
    print("Testing Gymnasium wrapper...")
    env = G1ReachingGymEnv()

    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation sample: {obs[:5]}...")

    print("\nTesting random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, terminated={terminated}, distance={info.get('distance_to_target', 0):.3f}m")

    env.close()
    print("\n[SUCCESS] Gymnasium wrapper test complete!")
