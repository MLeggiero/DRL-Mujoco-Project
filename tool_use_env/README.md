# Hammer Grasping Environment for Unitree G1 Robot

A MuJoCo-based reinforcement learning environment for training a Unitree G1 humanoid robot with manipulative hands to grasp and interact with a hammer on a table.

## Overview

This environment implements a **grasping task** where:
- The robot (G1 with hands) has a **fixed pelvis** (stable base)
- The right and left arms/hands are fully controllable
- A **hammer with realistic physics** is placed on a table
- The goal is to train the robot to grasp and manipulate the hammer using its hands

### Key Features

✓ **Full Physics Simulation**
- MuJoCo physics engine with implicit integration
- Hammer with heavier head and lighter handle
- Collision detection between hands and hammer
- Free-floating hammer dynamics

✓ **Robot Configuration**
- Unitree G1 robot with hands (full finger articulation)
- Fixed base (pelvis) for stable grasping
- 31+ controllable joints (arms, hands, torso)
- Both right and left hands for dual manipulation

✓ **Grasping Task Setup**
- Hammer on table in front of robot
- Multiple contact points (fingers and palm)
- Grasp detection via physics contact simulation
- Contact rewards for learning

## File Structure

```
tool_use_env/
├── hammer_grasp_scene.xml           # MuJoCo scene definition with robot, table, hammer
├── hammer_grasp_environment.py      # Core environment implementation
├── hammer_gym_wrapper.py            # Gymnasium wrapper for RL training
├── visualize_hammer_grasp.py        # Visualization script
├── test_scene_load.py               # Quick test to verify scene loads
├── assets/
│   ├── *.STL                        # Robot body meshes (copied from unitree_g1)
│   ├── source/
│   │   └── low_poly_hammer.obj      # Low-poly hammer mesh
│   └── textures/
│       ├── Texture__Albedo_Map.tga.png
│       ├── Texture__normals.tga.png
│       └── ... (texture files)
└── README.md                        # This file
```

## Physics Model Details

### Robot (Unitree G1 with Hands)

- **Base**: Fixed joint at pelvis
- **Legs**: 6 DOF per leg (locked in initial configuration)
- **Torso**: 3 DOF (waist yaw, roll, pitch)
- **Arms**: 7 DOF per arm (shoulders, elbows, wrists)
- **Hands**: 9 DOF per hand (thumbs, index, middle fingers)
  - **Total controllable**: ~31 joints

### Hammer

- **Head**: Heavy rectangular block (0.8 kg) at bottom
  - Size: 0.03m × 0.05m × 0.05m
  - Density: 800 kg/m³
  - Acts as the "impact" end

- **Handle**: Tapered cylinder (lighter material)
  - Lower section: 0.016m diameter, 0.08m length
  - Upper section: 0.012m diameter, 0.08m length
  - Density: 150-200 kg/m³

- **Total mass**: ~1.0 kg
- **Joint**: Free joint (6 DOF)
- **Friction**: 0.6 (head), 0.5 (handle)

### Table

- **Dimensions**: 1.6m × 1.2m × 0.66m (width × depth × height)
- **Material**: Wooden appearance (friction 1.0)
- **Position**: 0.6m from robot in x-direction
- **Hammer placement**: ~0.5m from robot

## Environment API

### Initialization

```python
from hammer_grasp_environment import HammerGraspEnv

env = HammerGraspEnv(
    scene_path="hammer_grasp_scene.xml",
    action_smoothing=0.3,          # EMA coefficient for action smoothing
    smoothness_weight=1.0,          # Penalty weight for jerky movements
    action_scale=0.5,              # Scale factor for joint commands
    sim_substeps=10,               # Physics steps per environment step
    use_hand_control=True          # Include hand/finger actuators
)
```

### Core Methods

#### `reset() -> np.ndarray`
Resets environment and returns initial observation.

```python
obs = env.reset()  # Shape: (40,)
```

#### `step(action: np.ndarray) -> (np.ndarray, float, bool, Dict)`
Executes one environment step.

```python
obs, reward, done, info = env.step(action)
```

**Action**: Continuous control commands for each actuator
- Shape: (n_actions,) where n_actions depends on configuration
- Range: [-1.0, 1.0] (automatically scaled and clipped)

**Observation** (40 dimensions):
- Right hand position (3)
- Left hand position (3)
- Hammer position (3)
- Hammer velocity (3)
- Arm joint positions (14)
- Arm joint velocities (14)

**Reward Components**:
- Distance-based: -hand_to_hammer_distance * 0.5
- Contact bonus: +0.5 per step when touching hammer
- Grasp bonus: +1.0 for stable grasp (>10 contact frames)
- Smoothness penalty: -action_smoothness * 0.01

### Gymnasium Wrapper

```python
from hammer_gym_wrapper import HammerGraspGymWrapper
import gymnasium as gym

env = HammerGraspGymWrapper()

obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

Compatible with **Stable-Baselines3**:
```python
from stable_baselines3 import PPO

env = HammerGraspGymWrapper()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Visualization

### Quick Scene Verification

```bash
cd tool_use_env
python test_scene_load.py
```

Outputs:
- Scene loading status
- Physics simulation test
- Body positions and contacts

### Interactive Visualization

```bash
cd tool_use_env
python visualize_hammer_grasp.py
```

**Controls** (in MuJoCo viewer):
- **Space**: Play/pause simulation
- **Right mouse drag**: Rotate camera
- **Scroll**: Zoom
- **'C'**: Toggle camera tracking
- **Right click drag on objects**: Apply forces

## Training

### Basic Training Example

```python
from hammer_gym_wrapper import HammerGraspGymWrapper
from stable_baselines3 import PPO
import numpy as np

# Create environment
env = HammerGraspGymWrapper(use_hand_control=True)

# Configure policy
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    verbose=1
)

# Train
model.learn(total_timesteps=500000)

# Save
model.save("hammer_grasping_ppo")
```

### Training Tips

1. **Start with reaching**: First train just hand positioning
2. **Gradually add hand control**: Then enable finger actuation
3. **Use curriculum learning**: Hammer starts close, gradually moves further
4. **Monitor contacts**: Use `info['contact_with_hammer']` for curriculum
5. **Increase episode length**: Start with 200 steps, increase to 800+

## Customization

### Disable Hand Control

For faster training or simpler policy:
```python
env = HammerGraspEnv(use_hand_control=False)
# Action space will be ~17 dimensions (arms + torso only)
```

### Modify Action Smoothing

For smoother or more reactive control:
```python
env = HammerGraspEnv(
    action_smoothing=0.5,      # More smoothing (0=max, 1=none)
    smoothness_weight=0.5      # Less penalty for jerky movements
)
```

### Change Reward Weights

Modify `_compute_reward()` in `hammer_grasp_environment.py`:
```python
# Increase distance reward importance
distance_reward = -hand_to_hammer_dist * 0.8  # Was 0.5

# Increase contact reward
contact_reward = 1.0  # Was 0.5
```

### Modify Hammer Properties

Edit `hammer_grasp_scene.xml`:

```xml
<!-- Hammer head mass and density -->
<inertial pos="0.0 0.0 -0.04" mass="1.2" diaginertia="0.001 0.001 0.001"/>
<geom name="hammer_head" type="box" size="0.015 0.025 0.025"
      density="900" friction="0.7" />

<!-- Hammer handle friction -->
<geom name="hammer_handle_lower" ... friction="0.6" density="250" />
```

## Troubleshooting

### Scene won't load: "File not found"
- Ensure all STL files are in `assets/` directory
- Check that `hammer_grasp_scene.xml` references correct meshdir
- Run from `tool_use_env` directory or adjust path

### "No bodies found" or "No actuators"
- Verify G1 robot meshes are copied to assets/
- Check that XML references mesh files by name (not full path)
- Check MuJoCo version compatibility

### Low reward / slow learning
- Increase grasp contact frames threshold in `_compute_reward()`
- Adjust reward weights for your specific task
- Start with longer episodes (max_episode_steps=2000)
- Use curriculum learning to gradually increase difficulty

### Simulation runs slow
- Reduce `sim_substeps` (default 10, try 5)
- Disable hand control: `use_hand_control=False`
- Use simpler collision geometries

## Next Steps

After verifying the environment works:

1. **Train a baseline**: Simple reaching task first
2. **Add object interaction**: Train grasp and lift tasks
3. **Multi-object**: Train with different objects
4. **Complex manipulation**: Tool use, assembly tasks
5. **Real robot deployment**: Transfer to physical G1

## Technical Details

### Physics Settings

- **Timestep**: 0.004 seconds (250 Hz)
- **Solver**: Implicit integration (implicitfast)
- **Iterations**: 5 solver iterations
- **Contact model**: Sliding friction with 1D constraint
- **Quaternion format**: WXYZ (MuJoCo standard)

### Coordinate System

- **X-axis**: Forward (robot facing +X)
- **Y-axis**: Left/right (left is +Y)
- **Z-axis**: Up

### Joint Ranges

All joint limits are preserved from the G1 URDF:
- Hip/shoulder pitch: ~±1.57 rad
- Hip/shoulder roll: ~±0.52 to ±2.97 rad
- Waist: ~±0.52 rad
- Elbow: ~±1.05 rad
- Wrist: ~±1.61 to ±1.97 rad

## References

- **MuJoCo**: https://mujoco.org/
- **Unitree G1**: https://unitreerobotics.com/
- **Gymnasium**: https://gymnasium.farama.org/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

## License

Based on Unitree G1 model and MuJoCo environments.
See original licenses for G1 URDF and STL files.
