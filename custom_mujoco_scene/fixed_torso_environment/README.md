# Unitree G1 Reaching Task - Reinforcement Learning

Professional RL training framework for teaching the Unitree G1 humanoid robot to perform reaching tasks using Stable-Baselines3 and MuJoCo.

## Overview

This project implements a reinforcement learning environment where the Unitree G1 humanoid robot learns to reach and touch target objects using its right arm while maintaining a stable base with locked legs.

### Key Features

- **Smooth Motion Control**: Action filtering with exponential moving average and velocity penalties
- **Stable Training**: Target KL divergence control, gradient clipping, and bounded exploration
- **Parallel Environments**: Multi-core CPU support for accelerated training
- **GPU Acceleration**: CUDA support for neural network training
- **Comprehensive Reward Shaping**: Exponential distance rewards, progress tracking, and proximity bonuses

## Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- MuJoCo 2.3+
- stable-baselines3
- gymnasium
- numpy
- torch (with CUDA support recommended)

## Quick Start

### Basic Training

```bash
python train_sb3.py --scene ../unitree_g1/g1_table_box_scene.xml
```

### Optimized Training (Recommended)

```bash
python train_sb3_improved.py --scene ../unitree_g1/g1_table_box_scene.xml
```

### Testing Trained Model

```bash
python train_sb3.py --test models/path/to/model --scene ../unitree_g1/g1_table_box_scene.xml
```

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timesteps` | 1,000,000 | Total training timesteps |
| `--lr` | 5e-4 | Learning rate |
| `--n-envs` | 4 | Number of parallel environments |
| `--device` | cuda | Training device (cuda/cpu/auto) |
| `--batch-size` | 256 | Minibatch size for updates |
| `--n-epochs` | 10 | PPO update epochs |

### Smoothness Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--action-smoothing` | 0.2 | EMA filtering coefficient (lower = smoother) |
| `--smoothness-weight` | 3.0 | Action change penalty weight |
| `--action-scale` | 0.35 | Actuator command scaling |
| `--sim-substeps` | 10 | Physics steps per RL step |

## Architecture

### Environment (`g1_rl_environment.py`)

- **Observation Space**: Joint positions/velocities, end-effector position, target location
- **Action Space**: Right arm (7 DOF) + torso (3 DOF) continuous control
- **Reward Function**: Distance minimization, progress tracking, smoothness penalties
- **Episode Termination**: Success (< 5cm) or timeout (400 steps)

### Gym Wrapper (`g1_gym_wrapper.py`)

Provides Gymnasium-compatible interface for Stable-Baselines3 integration.

### Training Scripts

- `train_sb3.py`: Configurable baseline training script
- `train_sb3_improved.py`: Optimized configuration for fast convergence and smooth motion

## Reward Structure

The reward function balances goal achievement with motion quality:

### Distance Reward
Combines exponential and linear terms: `-5*d + 10*exp(-5*d)`
Provides smooth gradient from far to near distances.

### Progress Reward
Continuous feedback: `50 * (prev_distance - current_distance)`
Encourages consistent approach to target with triple penalty for moving away.

### Proximity Bonuses (Cumulative)
- 20cm: +10
- 15cm: +20
- 10cm: +50
- 8cm: +100
- 6cm: +200
- < 5cm: +2000 (success)

### Smoothness Penalties
- Joint velocity: `-0.01 * sum(joint_vel²)`
- Action changes: `-3.0 * sum((action - prev_action)²)`
- Action magnitude: `-0.005 * sum(action²)`

## Training Results

Expected performance with optimized configuration:

- **Convergence**: First successes within 50k-100k steps
- **Success Rate**: 70-90% by 500k-1M steps
- **Training Time**: 15-20 minutes (1M steps, 4 parallel environments)
- **Stability Metrics**: KL divergence < 0.05, policy std < 5000

## Project Structure

```
fixed_torso_environment/
├── g1_rl_environment.py      # Core MuJoCo environment
├── g1_gym_wrapper.py          # Gymnasium wrapper
├── train_sb3.py               # Training script (configurable)
├── train_sb3_improved.py      # Optimized training configuration
├── visualize_policy.py        # Policy visualization tool
├── requirements.txt           # Python dependencies
├── docs/                      # Additional documentation
│   └── VISUALIZATION_GUIDE.md
├── scripts/                   # Utility scripts
└── models/                    # Trained model checkpoints
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
```

### Real-time Metrics

Training outputs:
- Episode rewards and lengths
- Success rate
- Distance to target
- KL divergence and policy statistics

## Advanced Usage

### Custom Reward Tuning

Modify `g1_rl_environment.py::_calculate_reward()` to adjust:
- Distance reward weights
- Proximity bonus thresholds
- Smoothness penalty strengths

### Parallel Environment Scaling

```bash
# Use 8 parallel environments (requires 8+ CPU cores)
python train_sb3_improved.py --n-envs 8

# Disable parallelization for debugging
python train_sb3.py --n-envs 1
```

### Hyperparameter Optimization

Key parameters for tuning:
- **Learning rate**: Balance between stability and convergence speed
- **Batch size**: Affects gradient estimate variance
- **Entropy coefficient**: Controls exploration magnitude
- **Target KL**: Prevents policy divergence

## Troubleshooting

### High KL Divergence (> 2.0)
- Reduce learning rate
- Decrease number of epochs
- Verify target_kl is enabled (0.02)

### Policy Not Converging
- Increase success bonus weight in reward function
- Check reward clipping limits
- Verify observation normalization is active

### Shaky or Jerky Motion
- Increase action smoothing (lower alpha value)
- Increase smoothness penalty weight
- Reduce action scaling factor
- Increase physics substeps

### Out of Memory
- Reduce number of parallel environments
- Decrease batch size
- Use smaller network architecture

## Visualization

```bash
./scripts/visualize.sh
```

This launches the MuJoCo viewer with the best trained model. See `docs/VISUALIZATION_GUIDE.md` for detailed controls and options.

## License

This project is provided for research and educational purposes.
