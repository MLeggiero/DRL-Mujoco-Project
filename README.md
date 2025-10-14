# Unitree G1 Reinforcement Learning for Reach and Touch Tasks

A reinforcement learning framework for training the Unitree G1 humanoid robot to perform reaching and manipulation tasks in MuJoCo simulation.

## Overview

This project implements a custom RL environment and training pipeline for the Unitree G1 robot to learn table-top manipulation tasks. The robot learns to reach and touch target objects placed on a table using simple policy gradient methods.

## Features

- **Custom MuJoCo Environment**: G1 robot with table and manipulatable objects
- **RL Training Pipeline**: Simple policy gradient implementation for reaching tasks
- **Flexible Scene Configuration**: Easy-to-modify XML-based scene setup
- **Visual Monitoring**: Real-time training progress visualization
- **Interactive Launchers**: Multiple viewing and testing modes

## Prerequisites

- Python 3.10 or later
- MuJoCo 3.3.6 or later
- Ubuntu 20.04+ (via WSL2 for Windows users)

## Installation

### 1. Set Up Environment

```bash
# Create project directory
mkdir -p ~/mujoco_projects
cd ~/mujoco_projects

# Create virtual environment
python3 -m venv mujoco_env
source mujoco_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install mujoco numpy matplotlib
```

### 2. Clone Repository

```bash
cd ~/mujoco_projects
# Clone your repository here
```

### 3. Download MuJoCo Menagerie Models

```bash
# Clone the MuJoCo Menagerie for G1 model files
git clone https://github.com/google-deepmind/mujoco_menagerie.git

# Copy G1 files to project
cp -r mujoco_menagerie/unitree_g1 .
```

## File Structure

```
mujoco_projects/
├── unitree_g1/
│   ├── g1.xml                          # Base G1 robot model
│   ├── g1_table_box_scene.xml          # Custom scene with table and objects
│   └── assets/                         # G1 mesh and texture files
├── custom_scene_launcher.py            # Interactive scene viewer
├── g1_rl_environment.py                # RL environment implementation
├── g1_training_script.py               # Training script
└── mujoco_env/                         # Python virtual environment
```

## Usage

### Viewing the Scene

Launch the interactive scene viewer to visualize the G1 robot and environment:

```bash
source mujoco_env/bin/activate
python custom_scene_launcher.py
```

**Controls:**
- Mouse drag: Rotate camera
- Shift + Mouse: Pan camera
- Ctrl + Mouse: Zoom
- Space: Pause/resume physics
- ESC: Close viewer

### Testing the RL Environment

Verify the RL environment is properly configured:

```bash
python g1_training_script.py
# Select option 3: "Test environment only"
```

This will:
- Load the scene and robot model
- Reset the environment
- Run random actions to test functionality
- Display distance-to-target metrics

### Training the Robot

Start RL training to teach the robot reaching behaviors:

```bash
python g1_training_script.py
```

**Training Options:**
1. **Full training** (1000 episodes, ~20-30 minutes)
2. **Quick test** (100 episodes, ~3-5 minutes)
3. **Environment test only**

**Training Output:**
- Real-time progress updates every 10 episodes
- Success rate monitoring
- Automatic plot generation (`g1_training_progress.png`)
- Policy weights saved to `g1_training_results.pkl`

### Example Training Session

```bash
$ python g1_training_script.py

G1 Reach-Touch RL Training
============================================================
Choose training mode:
1. Full training (1000 episodes)
2. Quick test (100 episodes)
3. Test environment only
Enter choice (1-3): 2

Starting quick test training...
Creating policy with obs_dim=27, action_dim=8
Episode    0: Reward= -45.2, Length= 200.0, Success=  0.0%
Episode   10: Reward= -38.7, Length= 195.0, Success=  5.0%
Episode   20: Reward= -32.1, Length= 180.0, Success= 15.0%
...
```

## Environment Details

### Observation Space

The environment provides:
- **Robot proprioception**: Joint positions and velocities (20 DoF)
- **End-effector position**: 3D position of the robot's hand
- **Target position**: 3D position of target object
- **Distance to target**: Euclidean distance metric

### Action Space

- **Dimension**: 8 (arm joint torques)
- **Range**: [-1, 1] (normalized)
- **Control frequency**: 200 Hz (5 simulation steps per RL step)

### Reward Function

```
reward = distance_reward + progress_reward + success_reward + time_penalty
```

- **Distance reward**: -10.0 * distance (encourages proximity)
- **Progress reward**: 50.0 * (previous_distance - current_distance)
- **Success reward**: 100.0 (when within 8cm of target)
- **Time penalty**: -0.1 per step (encourages efficiency)

### Task Objectives

The robot must learn to:
1. Reach toward randomly selected target objects
2. Minimize distance between end-effector and target
3. Complete tasks efficiently (fewer steps)

**Target Objects:**
- Red box (0.04m cube)
- Blue cylinder (0.05m radius, 0.1m height)
- Blue cylinder 2 (smaller variant)
- Green cone (ellipsoid shape)

## Customization

### Modifying the Scene

Edit `unitree_g1/g1_table_box_scene.xml` to:
- Add new objects
- Change table dimensions
- Adjust lighting
- Modify robot starting position

### Adjusting Training Parameters

In `g1_training_script.py`:

```python
class G1Trainer:
    def __init__(self):
        # Modify these parameters
        self.max_episodes = 1000          # Total training episodes
        self.max_steps_per_episode = 200  # Steps per episode
        self.learning_rate = 0.001        # Policy learning rate
```

### Changing Robot Position

In `custom_scene_launcher.py`, modify:

```python
def setup_model_for_viewing(model, data, robot_position=None):
    if robot_position is None:
        robot_position = [-0.1, 0.0, 0.8]  # [x, y, z] in meters
```

## Results and Visualization

After training, the following files are generated:

- **g1_training_results.pkl**: Complete training data and policy weights
- **g1_training_progress.png**: Training curves showing:
  - Episode rewards over time
  - Episode lengths over time
  - Moving averages for trend analysis

## Troubleshooting

### Scene File Not Found

```bash
ERROR: Scene file not found at unitree_g1/g1_table_box_scene.xml
```

**Solution**: Ensure `g1_table_box_scene.xml` is in the `unitree_g1/` directory.

### No Arm Actuators Found

```bash
WARNING: No arm actuators found - using first 8 actuators
```

**Solution**: This is normal. The code will use the first 8 actuators as a fallback.

### Import Error

```bash
Error importing G1ReachTouchEnv: No module named 'g1_rl_environment'
```

**Solution**: Ensure all Python files are in the same directory and you're running from that directory.

### MuJoCo Viewer is Blank

Press the **Home** key to reset the camera view, then try zooming out with Ctrl + Mouse wheel.

## Limitations and Future Work

### Current Limitations

- **Simple policy architecture**: Linear policy with single hidden layer
- **No vision**: Relies on proprioceptive feedback only
- **Basic reward shaping**: Distance-based rewards only
- **Single arm control**: Right arm primarily used

### Planned Improvements

- [ ] Implement proper PPO/SAC algorithms
- [ ] Add camera-based visual observations
- [ ] Implement grasping and object manipulation
- [ ] Multi-object sequencing tasks
- [ ] Sim-to-real transfer preparation
- [ ] Integration with MuJoCo Playground framework

## Performance Benchmarks

**Hardware Tested:**
- CPU: Intel i7 or equivalent
- RAM: 16GB minimum
- GPU: Not required (CPU-only training)

**Expected Training Time:**
- 100 episodes: 3-5 minutes
- 1000 episodes: 20-30 minutes

**Expected Results:**
- Initial success rate: ~0%
- After 50 episodes: 5-15%
- After 200 episodes: 20-40%
- After 500+ episodes: 40-60%

*Note: Success rates depend on random initialization and may vary*

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{unitree_g1_rl,
  title={Unitree G1 Reinforcement Learning for Manipulation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/unitree-g1-rl}
}
```

## License

This project uses models from the MuJoCo Menagerie, which are subject to their respective licenses. The Unitree G1 model is licensed under BSD-3-Clause by Unitree Robotics.

## Acknowledgments

- **MuJoCo**: Physics simulation engine by DeepMind
- **MuJoCo Menagerie**: High-quality robot models by Google DeepMind
- **Unitree Robotics**: G1 humanoid robot and model files

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact [your email/contact info].

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Unitree G1 Robot](https://www.unitree.com/g1)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
