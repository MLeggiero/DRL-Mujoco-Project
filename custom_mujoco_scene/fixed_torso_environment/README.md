# G1 Robot Reaching Task - Reinforcement Learning

Train a Unitree G1 robot to reach and touch target objects using deep reinforcement learning with PPO (Proximal Policy Optimization).

## Quick Start

### 1. Installation

```bash
./scripts/install_dependencies.sh
```

This installs:
- MuJoCo physics simulator
- Stable-Baselines3 (PPO implementation)
- Gymnasium (RL environment interface)
- TensorBoard (training visualization)
- NumPy, Matplotlib, SciPy

### 2. Train a Model

**Recommended (500k timesteps, optimized hyperparameters):**
```bash
python train_sb3_improved.py
```

**Standard (100k timesteps, baseline):**
```bash
./scripts/train_sb3.sh
```

Training will:
- Save checkpoints every 5k steps to `./models/`
- Evaluate every 2.5k steps
- Track best model automatically
- Log to TensorBoard in `./logs/`

### 3. Visualize Trained Policy

```bash
./scripts/visualize.sh
```

This launches MuJoCo viewer and runs 5 episodes with the latest trained model.

**Manual visualization:**
```bash
# List available models
python visualize_policy.py --list-models

# Visualize specific model
python visualize_policy.py --model ./models/g1_ppo_*/best_model/best_model

# Visualize with options
python visualize_policy.py \
  --model ./models/g1_ppo_*/best_model/best_model \
  --episodes 10 \
  --slow \
  --stochastic
```

### 4. Monitor Training

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

## Environment Overview

**Task:** Control the G1 robot's right arm to reach and touch a red box on a table.

**Observations:**
- Joint positions (right arm + torso)
- Joint velocities
- End effector (hand) position
- Target position
- Hand-to-target vector

**Actions:**
- Continuous torques for right arm and torso joints
- Scaled to `[-0.01, 0.01]` for stability (ultra-gentle control)

**Rewards:**
- Distance to target: `-distance`
- Progress bonus: `10.0 * (prev_distance - current_distance)`
- Proximity bonuses:
  - Within 30cm: +1.0
  - Within 15cm: +3.0
  - Within 5cm (success): +10.0
- Time penalty: -0.001 per step

**Episode Termination:**
- Success: Hand within 5cm of target
- Max steps: 300 steps
- Instability: Robot falls or simulation becomes unstable

## Key Files

### Core Environment
- `g1_rl_environment.py` - Main RL environment implementation
- `g1_gym_wrapper.py` - Gymnasium wrapper for Stable-Baselines3 compatibility

### Training Scripts
- `train_sb3.py` - Standard SB3 training (100k steps, baseline)
- `train_sb3_improved.py` - Improved training (500k steps, optimized hyperparameters)
- `train_with_brax.py` - GPU-accelerated training with Brax (optional)
- `train_auto.py` - Auto-detect hardware and choose best training method

### Visualization
- `visualize_policy.py` - MuJoCo viewer for trained policies
- `test_environment.py` - Test environment setup and observations
- `test_reachability.py` - Verify target is physically reachable

### Utilities
- `scripts/install_dependencies.sh` - Install all required packages
- `scripts/visualize.sh` - Quick visualization launcher
- `scripts/train_sb3.sh` - Quick training launcher

## Configuration Details

### Current Environment Settings

**Action Scaling:** 0.01 (ultra-gentle torques for stability)
- Prevents DOF 19 instability warnings
- Slower movements but more controlled
- Requires longer training (500k+ timesteps)

**Episode Length:** 300 steps
- Reduced from 1000 for faster learning
- Robot has ~15 seconds to reach target (50 Hz control)

**Fixed Base:** Base position and legs are locked in place
- Simplifies learning problem
- Focuses training on arm control only
- Uses PD control to hold leg joints

### Training Hyperparameters (Improved Config)

```python
{
    "total_timesteps": 500_000,
    "learning_rate": 5e-4,
    "n_steps": 1024,
    "batch_size": 128,
    "n_epochs": 20,
    "gamma": 0.98,
    "save_freq": 5_000,
    "eval_freq": 2_500,
}
```

## Training Progress Expectations

**Good indicators:**
- Reward improving from -200 to -50 or better
- Distance decreasing from 0.5m to < 0.1m
- Success rate > 20% by end of training
- Minimum distance achieved < 0.05m

**Poor indicators:**
- Reward stuck around -300
- Distance not decreasing
- Success rate 0%
- DOF 19 instability warnings (action scaling too high)

## Troubleshooting

### DOF 19 Instability Warnings

**Problem:** `WARNING: Nan, Inf or huge value in QACC at DOF 19`

**Cause:** Action scaling too high, causing floating base to become unstable

**Solution:** Reduce action scaling in `g1_rl_environment.py`:
```python
self.data.ctrl[actuator_id] = action[i] * 0.01  # Reduce from 0.1 → 0.01
```

### Robot Moves Erratically

**Problem:** Robot flailing, moving away from target

**Causes:**
1. Visualizing old model trained before fixes
2. Action scaling too high
3. Not enough training

**Solutions:**
1. Retrain with current environment version
2. Reduce action scaling to 0.01
3. Train for 500k+ timesteps

### Training Not Improving

**Problem:** Reward stuck, no progress after many timesteps

**Possible causes:**
1. Action scaling too low (robot can't move enough)
2. Time penalty too high (dominates reward signal)
3. Episode length too short (not enough time to learn)

**Solutions:**
1. Increase action scaling slightly (0.01 → 0.02)
2. Verify time penalty is -0.001 (not -0.01)
3. Increase max_episode_steps to 500

### Viewer Doesn't Open

**Problem:** MuJoCo viewer window doesn't appear

**Solutions:**
```bash
# Check display is available
echo $DISPLAY

# Install display libraries (Linux)
sudo apt-get install libgl1-mesa-glx

# Run locally (not over SSH)
```

## Advanced Usage

### Custom Training Configuration

```python
from train_sb3 import train_g1_reaching

train_g1_reaching(
    scene_path="../unitree_g1/g1_table_box_scene.xml",
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    save_freq=10_000,
    eval_freq=5_000,
)
```

### Testing Reachability

```bash
python test_reachability.py
```

Tests if target is physically reachable with random actions. Useful for verifying workspace constraints.

### Comparing Models

```bash
# Best model
python visualize_policy.py --model ./models/run1/best_model/best_model --episodes 5

# Final model
python visualize_policy.py --model ./models/run1/final_model --episodes 5

# Random baseline
python visualize_policy.py --random --episodes 5
```

## Migration from Custom PPO

This project originally used a custom PPO implementation that had numerical instability issues. We migrated to Stable-Baselines3 for:
- Better numerical stability (no NaN crashes)
- Industry-standard implementation
- Built-in observation/reward normalization
- TensorBoard integration
- Checkpoint management
- Easier debugging and monitoring

See `MIGRATION_GUIDE.md` for technical details.

## Documentation Files

- `README.md` (this file) - Main documentation
- `VISUALIZATION_GUIDE.md` - Detailed visualization instructions and MuJoCo viewer controls
- `TRAINING_IMPROVEMENTS.md` - Deep dive into training optimization techniques
- `QUICK_START_IMPROVEMENTS.md` - Step-by-step fixes for common training issues
- `MIGRATION_GUIDE.md` - Custom PPO to Stable-Baselines3 migration details
- `CPU_TRAINING_GUIDE.md` - CPU-specific training optimizations

## Project Structure

```
fixed_torso_environment/
├── g1_rl_environment.py          # Core RL environment
├── g1_gym_wrapper.py              # Gymnasium wrapper
├── train_sb3.py                   # Standard training script
├── train_sb3_improved.py          # Improved training config
├── visualize_policy.py            # Visualization tool
├── test_environment.py            # Environment testing
├── test_reachability.py           # Reachability testing
├── requirements.txt               # Python dependencies
├── scripts/                       # Shell scripts
│   ├── install_dependencies.sh    # Dependency installation
│   ├── train_sb3.sh               # Training launcher
│   └── visualize.sh               # Visualization launcher
├── models/                        # Trained models (generated)
├── logs/                          # TensorBoard logs (generated)
└── docs/                          # Additional documentation
```

## Contributing

When modifying the environment or training code:
1. Test changes with `python test_environment.py`
2. Verify reachability with `python test_reachability.py`
3. Train for at least 100k steps to validate improvements
4. Visualize results to confirm behavior

## License

This project uses MuJoCo (Apache 2.0), Stable-Baselines3 (MIT), and Gymnasium (MIT).
