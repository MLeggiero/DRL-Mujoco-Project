#!/usr/bin/env python3
"""
CNN and hybrid policies for RGBD-based hammer grasping environment.

Provides various network architectures for learning from vision and proprioception:
- Pure CNN (vision-only)
- CNN + MLP (hybrid vision + proprioception)
- Dual-stream CNN (RGB + depth separate processing)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    """Simple CNN for extracting features from RGBD images."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        """Initialize CNN feature extractor.

        Args:
            observation_space: Observation space (image shape)
            features_dim: Dimension of output features
        """
        super().__init__(observation_space, features_dim)

        # Assume input shape is (H, W, C)
        h, w, c = observation_space.shape

        # Define CNN layers
        self.cnn = nn.Sequential(
            # Input: (C, H, W) after permutation
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Output: (32, (H-8+0)/4+1, (W-8+0)/4+1)

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Output: (64, ...)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_output = self.cnn(dummy_input)
            cnn_output_size = cnn_output.reshape(1, -1).shape[1]

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN.

        Args:
            observations: (batch_size, H, W, C)

        Returns:
            Features: (batch_size, features_dim)
        """
        # Permute to (batch_size, C, H, W) for CNN
        x = observations.permute(0, 3, 1, 2).contiguous()

        # CNN forward
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)

        # FC forward
        x = self.fc(x)

        return x


class DualStreamCNNFeaturesExtractor(BaseFeaturesExtractor):
    """Dual-stream CNN processing RGB and depth separately."""

    def __init__(self,
                 observation_space: Union[spaces.Dict, spaces.Box],
                 features_dim: int = 256):
        """Initialize dual-stream CNN.

        Args:
            observation_space: Dict space with 'rgb' and 'depth'
            features_dim: Output feature dimension
        """
        if isinstance(observation_space, spaces.Dict):
            super().__init__(observation_space, features_dim)
            self.is_dict = True
            rgb_space = observation_space['rgb']
            depth_space = observation_space['depth']
        else:
            # Assume stacked RGBD
            super().__init__(observation_space, features_dim)
            self.is_dict = False
            h, w, c = observation_space.shape
            rgb_space = spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
            depth_space = spaces.Box(0, 1, shape=(h, w), dtype=np.float32)

        h_rgb, w_rgb, c_rgb = rgb_space.shape
        h_depth, w_depth = depth_space.shape

        # RGB stream
        self.rgb_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # Depth stream
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # Calculate flattened sizes
        with torch.no_grad():
            rgb_dummy = torch.zeros(1, 3, h_rgb, w_rgb)
            rgb_output = self.rgb_cnn(rgb_dummy)
            rgb_size = rgb_output.reshape(1, -1).shape[1]

            depth_dummy = torch.zeros(1, 1, h_depth, w_depth)
            depth_output = self.depth_cnn(depth_dummy)
            depth_size = depth_output.reshape(1, -1).shape[1]

        # Fusion layer
        combined_size = rgb_size + depth_size
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Union[torch.Tensor, dict]) -> torch.Tensor:
        """Forward pass through dual-stream CNN.

        Args:
            observations: Stacked RGBD tensor or dict with 'rgb'/'depth'

        Returns:
            Features: (batch_size, features_dim)
        """
        if isinstance(observations, dict):
            rgb = observations['rgb'].permute(0, 3, 1, 2).contiguous()
            depth = observations['depth'].unsqueeze(1)
        else:
            # Split stacked RGBD into RGB (3 channels) and depth (1 channel)
            # Assuming (batch, H, W, 4*stack_frames) stacked RGBD
            batch_size, h, w, c = observations.shape
            stack = c // 4

            # Separate RGB and depth channels
            rgb_list = []
            depth_list = []
            for i in range(stack):
                frame = observations[:, :, :, i*4:(i+1)*4]
                rgb_list.append(frame[:, :, :, :3])
                depth_list.append(frame[:, :, :, 3])

            # Average or concatenate stacked frames
            rgb = torch.stack(rgb_list, dim=3).mean(dim=3)  # (batch, H, W, 3)
            rgb = rgb.permute(0, 3, 1, 2).contiguous()

            depth = torch.stack(depth_list, dim=3).mean(dim=3)  # (batch, H, W)
            depth = depth.unsqueeze(1)

        # RGB stream
        rgb_features = self.rgb_cnn(rgb)
        rgb_features = rgb_features.reshape(rgb_features.shape[0], -1)

        # Depth stream
        depth_features = self.depth_cnn(depth)
        depth_features = depth_features.reshape(depth_features.shape[0], -1)

        # Concatenate and fuse
        combined = torch.cat([rgb_features, depth_features], dim=1)
        features = self.fusion(combined)

        return features


class HybridFeaturesExtractor(BaseFeaturesExtractor):
    """Hybrid extractor combining vision and proprioceptive features."""

    def __init__(self,
                 observation_space: spaces.Dict,
                 features_dim: int = 256,
                 vision_dim: int = 128,
                 proprio_dim: int = 64):
        """Initialize hybrid feature extractor.

        Args:
            observation_space: Dict space with 'rgb', 'depth', 'proprioceptive'
            features_dim: Total output dimension
            vision_dim: Vision features dimension
            proprio_dim: Proprioceptive features dimension
        """
        super().__init__(observation_space, features_dim)

        rgb_space = observation_space['rgb']
        h, w = rgb_space.shape[:2]

        # Vision CNN
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, h, w)
            cnn_out = self.vision_cnn(dummy)
            cnn_size = cnn_out.reshape(1, -1).shape[1]

        self.vision_fc = nn.Sequential(
            nn.Linear(cnn_size, vision_dim),
            nn.ReLU()
        )

        # Proprioceptive MLP
        self.proprio_fc = nn.Sequential(
            nn.Linear(28, 64),  # 28-dim proprioceptive input
            nn.ReLU(),
            nn.Linear(64, proprio_dim),
            nn.ReLU()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + proprio_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """Forward pass through hybrid network.

        Args:
            observations: Dict with 'rgb' and 'proprioceptive'

        Returns:
            Features: (batch_size, features_dim)
        """
        # Process vision
        rgb = observations['rgb'].permute(0, 3, 1, 2).contiguous()
        vision_features = self.vision_cnn(rgb)
        vision_features = vision_features.reshape(vision_features.shape[0], -1)
        vision_features = self.vision_fc(vision_features)

        # Process proprioception
        proprio = observations['proprioceptive']
        proprio_features = self.proprio_fc(proprio)

        # Fuse
        combined = torch.cat([vision_features, proprio_features], dim=1)
        features = self.fusion(combined)

        return features


# Usage example for training with Stable-Baselines3:
"""
from stable_baselines3 import PPO
from hammer_rgbd_gym_wrapper import HammerRGBDGymWrapper

# Create environment
env = HammerRGBDGymWrapper(observation_mode='rgbd_stacked')

# Create model with custom feature extractor
model = PPO(
    "CnnPolicy",  # Uses CNNFeaturesExtractor
    env,
    policy_kwargs={
        "features_extractor_class": CNNFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256}
    },
    verbose=1
)

model.learn(total_timesteps=500000)

# For hybrid (vision + proprioception):
env = HammerRGBDGymWrapper(observation_mode='rgbd_raw')

model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class": HybridFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 512,
            "vision_dim": 256,
            "proprio_dim": 128
        }
    },
    verbose=1
)
"""

if __name__ == "__main__":
    print("Testing CNN policies...")

    # Test CNN extractor
    print("\n=== Testing CNNFeaturesExtractor ===")
    obs_space = spaces.Box(0, 255, shape=(240, 320, 16), dtype=np.float32)  # 4 stacked RGBD
    cnn_extractor = CNNFeaturesExtractor(obs_space, features_dim=256)

    dummy_obs = torch.randn(4, 240, 320, 16)
    features = cnn_extractor(dummy_obs)
    print(f"Input shape: {dummy_obs.shape}")
    print(f"Output shape: {features.shape}")

    # Test Dual-stream extractor
    print("\n=== Testing DualStreamCNNFeaturesExtractor ===")
    obs_space_dict = spaces.Dict({
        'rgb': spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8),
        'depth': spaces.Box(0, 1, shape=(240, 320), dtype=np.float32),
    })
    dual_extractor = DualStreamCNNFeaturesExtractor(obs_space_dict, features_dim=256)

    dummy_dict = {
        'rgb': torch.randint(0, 255, (4, 240, 320, 3), dtype=torch.uint8).float(),
        'depth': torch.rand(4, 240, 320)
    }
    features = dual_extractor(dummy_dict)
    print(f"Output shape: {features.shape}")

    # Test Hybrid extractor
    print("\n=== Testing HybridFeaturesExtractor ===")
    obs_space_hybrid = spaces.Dict({
        'rgb': spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8),
        'depth': spaces.Box(0, 1, shape=(240, 320), dtype=np.float32),
        'proprioceptive': spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)
    })
    hybrid_extractor = HybridFeaturesExtractor(obs_space_hybrid, features_dim=512)

    dummy_hybrid = {
        'rgb': torch.randint(0, 255, (4, 240, 320, 3), dtype=torch.uint8).float(),
        'depth': torch.rand(4, 240, 320),
        'proprioceptive': torch.randn(4, 28)
    }
    features = hybrid_extractor(dummy_hybrid)
    print(f"Output shape: {features.shape}")

    print("\nAll policy tests passed!")
