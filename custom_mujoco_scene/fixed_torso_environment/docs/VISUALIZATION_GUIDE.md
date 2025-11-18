# Visualization Guide

Learn how to visualize your trained policies in the MuJoCo viewer to see what the robot has learned.

## Quick Start

### Option 1: Visualize Latest Model (Easiest)

```bash
./visualize.sh
```

This automatically finds your most recent best model and visualizes it.

### Option 2: Visualize Specific Model

```bash
# List available models
python visualize_policy.py --list-models

# Visualize a specific model
python visualize_policy.py --model ./models/g1_ppo_20251117_133846/best_model/best_model
```

### Option 3: Visualize Random Policy (Baseline)

```bash
python visualize_policy.py --random
```

Useful for comparing against untrained behavior.

## Detailed Usage

### Basic Visualization

```bash
python visualize_policy.py --model <path_to_model>
```

This will:
- Load the trained model
- Launch MuJoCo viewer window
- Run 5 episodes
- Show statistics

### Visualization Options

**Number of episodes:**
```bash
python visualize_policy.py --model <path> --episodes 10
```

**Slow motion (easier to see):**
```bash
python visualize_policy.py --model <path> --slow
```

**Stochastic policy (with exploration):**
```bash
python visualize_policy.py --model <path> --stochastic
```

**Combine options:**
```bash
python visualize_policy.py \
  --model ./models/g1_ppo_20251117_133846/best_model/best_model \
  --episodes 10 \
  --slow \
  --stochastic
```

## MuJoCo Viewer Controls

### Camera Controls
- **Left Mouse Drag**: Rotate camera around center
- **Right Mouse Drag**: Pan camera (translate)
- **Mouse Wheel**: Zoom in/out
- **Double Click**: Select and center on a body

### Playback Controls
- **Space**: Pause/Resume simulation
- **Backspace**: Reset simulation
- **Page Up/Down**: Adjust simulation speed
- **Esc**: Close viewer

### View Options
- **Tab**: Cycle through rendering modes
- **F1-F5**: Toggle visualization overlays
  - F1: Contact forces
  - F2: Joint axes
  - F3: Center of mass
  - F4: Bounding boxes
  - F5: Constraint forces

## Understanding the Visualization

### What to Look For

**Good Policy Indicators:**
- Robot reaches toward target smoothly
- Hand gets close to target (< 10cm)
- Movements are purposeful, not random
- Episode ends early (success)
- Final distance < 5cm

**Poor Policy Indicators:**
- Random-looking movements
- Hand moves away from target
- Jerky, unstable motions
- Always reaches max episode length
- Distance stays > 50cm

### Statistics Printed

During each episode:
```
Step 050 | Reward:   -45.32 | Distance: 0.234m | Min: 0.189m
```
- **Reward**: Cumulative reward (higher is better)
- **Distance**: Current hand-to-target distance
- **Min**: Minimum distance achieved so far

After each episode:
```
Episode 1 Complete:
  Total Reward: -123.45
  Steps: 234
  Final Distance: 0.156m
  Minimum Distance: 0.089m
  Success: NO
```

After all episodes:
```
OVERALL STATISTICS
Average Reward: -98.23 ± 12.34
Average Distance: 0.145m ± 0.023m
Success Rate: 20.0% (1/5)
Best Distance: 0.045m
```

## Comparing Models

### Compare Best vs Final Model

```bash
# Visualize best model
python visualize_policy.py \
  --model ./models/g1_ppo_20251117_133846/best_model/best_model \
  --episodes 5

# Visualize final model
python visualize_policy.py \
  --model ./models/g1_ppo_20251117_133846/final_model \
  --episodes 5
```

### Compare Against Random Baseline

```bash
# Trained policy
python visualize_policy.py --model <path> --episodes 5

# Random policy
python visualize_policy.py --random --episodes 5
```

### Compare Before/After Training Improvements

```bash
# Old model (before quick fixes)
python visualize_policy.py \
  --model ./models/g1_ppo_20251117_133846/best_model/best_model

# New model (after quick fixes)
python visualize_policy.py \
  --model ./models/g1_ppo_20251117_*/best_model/best_model
```

## Recording Video

To record a video of your policy:

### Option 1: Screen Recording
Use your OS screen recording:
- **Linux**: SimpleScreenRecorder, OBS Studio
- **Mac**: QuickTime, Cmd+Shift+5
- **Windows**: OBS Studio, Win+G

### Option 2: Programmatic Recording

Add to `visualize_policy.py`:
```python
# At the viewer loop
frames = []
for step in range(max_steps):
    # ... step environment ...

    # Capture frame
    renderer = mujoco.Renderer(base_env.model, height=480, width=640)
    renderer.update_scene(base_env.data)
    frame = renderer.render()
    frames.append(frame)

# Save video
import imageio
imageio.mimsave('policy_video.mp4', frames, fps=30)
```

## Troubleshooting

### Viewer doesn't open
- Check display is available: `echo $DISPLAY`
- Try running in local terminal, not SSH
- Install display libraries: `sudo apt-get install libgl1-mesa-glx`

### Model not found
```bash
# List available models
python visualize_policy.py --list-models

# Check path exists
ls ./models/g1_ppo_*/best_model/best_model.zip
```

### Robot moves erratically
- This is normal for early training
- Try later checkpoints or final model
- Compare with random policy to see if learning occurred

### Simulation is too fast
```bash
# Use slow motion flag
python visualize_policy.py --model <path> --slow

# Or press Page Down in viewer to slow down
```

### Normalization stats not found
Warning: `vecnormalize.pkl not found`

This is OK - model will still run but might perform worse. The visualization script handles this automatically.

## Example Workflow

### Full Visualization Workflow

```bash
# 1. List available models
python visualize_policy.py --list-models

# 2. Visualize best model (slow motion)
python visualize_policy.py \
  --model ./models/g1_ppo_20251117_133846/best_model/best_model \
  --slow \
  --episodes 10

# 3. Compare with random baseline
python visualize_policy.py --random --episodes 5

# 4. Check final model
python visualize_policy.py \
  --model ./models/g1_ppo_20251117_133846/final_model \
  --episodes 5
```

### Iterative Training & Visualization

```bash
# Train
python train_sb3.py --timesteps 100000

# Visualize
./visualize.sh

# If not good, apply fixes and retrain
./apply_quick_fixes.sh
python train_sb3_improved.py

# Visualize improved model
./visualize.sh
```

## Tips for Better Visualization

### 1. Use Slow Motion for Analysis
```bash
python visualize_policy.py --model <path> --slow
```
Makes it easier to see what the robot is doing.

### 2. Run Multiple Episodes
```bash
python visualize_policy.py --model <path> --episodes 10
```
See consistency across different starting positions.

### 3. Compare Deterministic vs Stochastic
```bash
# Deterministic (what it learned)
python visualize_policy.py --model <path>

# Stochastic (with exploration)
python visualize_policy.py --model <path> --stochastic
```

### 4. Watch Early vs Late Training
Visualize checkpoints to see learning progression:
```bash
# Early training (10k steps)
python visualize_policy.py --model ./models/.../g1_ppo_checkpoint_10000_steps

# Mid training (50k steps)
python visualize_policy.py --model ./models/.../g1_ppo_checkpoint_50000_steps

# Final (100k steps)
python visualize_policy.py --model ./models/.../best_model/best_model
```

## Common Observations

### What Different Behaviors Mean

**Robot reaches but misses:**
- Learning is happening but needs more training
- Try longer training or better reward shaping

**Robot moves randomly:**
- Not enough training
- Reward signal too weak
- Check that model loaded correctly

**Robot gets close (10-20cm) but not touching:**
- Good progress!
- Increase proximity rewards
- Train longer

**Robot touches target consistently:**
- Success! Policy is well-trained
- Consider making task harder (smaller target, randomization)

**Robot doesn't move:**
- Model might be overly conservative
- Increase entropy coefficient
- Check observation normalization

## Next Steps After Visualization

Based on what you see:

**If policy looks random:**
- Train much longer (500k+ steps)
- Apply quick fixes (episode length, time penalty)
- Check reward function

**If policy gets close but not quite:**
- Increase proximity rewards
- Reduce time penalty
- Train longer (300k+ steps)

**If policy is successful:**
- Add task difficulty (target randomization)
- Try more complex tasks
- Deploy to real robot!

## Files

- `visualize_policy.py` - Main visualization script
- `visualize.sh` - Quick launcher for latest model
- `g1_gym_wrapper.py` - Environment wrapper
- `g1_rl_environment.py` - Core environment

## See Also

- [README_SB3.md](README_SB3.md) - Training documentation
- [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) - How to improve results
- [QUICK_START_IMPROVEMENTS.md](QUICK_START_IMPROVEMENTS.md) - Quick fixes guide
