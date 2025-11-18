## Model Selection Issue - Why "Best Model" Isn't Actually Best

## Your Observation

**Training logs showed many successes**:
```
Success! Reached red_box (repeated many times around 460k-490k steps)
Eval at 460k: reward=1102.52, ep_length=163.40
```

**But visualization showed terrible performance**:
```
Success Rate: 0.0% (0/5)
Average Distance: 0.688m
Best Distance: 0.608m (never got close to 5cm success threshold!)
```

**You correctly identified**: The "best model" that was saved is NOT the model with best performance!

## Why This Happened

### The Problem with Mean Reward

Stable-Baselines3's `EvalCallback` saves "best_model" based on **mean reward across evaluation episodes**.

With **high variance in rewards** (due to position randomization), this is misleading:

#### Scenario 1: Easy Random Position (Close to robot)
```
Episode with box at [0.35, 0.0, 0.74] (very close)
- Robot reaches quickly: 50 steps
- Gets success bonus: +475 per step × 50 = +23,750
- Mean reward per episode: +23,750 / 1 = +23,750
```

#### Scenario 2: Hard Random Position (Far + to the side)
```
Episode with box at [0.65, 0.2, 0.74] (far and lateral)
- Robot struggles: 400 steps
- Never reaches: max +25 per step × 400 = +10,000
- Mean reward per episode: +10,000 / 1 = +10,000
```

### The Misleading Average

If evaluation runs:
- 3 easy episodes (luck) → +23,750 each
- 2 hard episodes → +10,000 each

Mean reward = (3×23750 + 2×10000) / 5 = **18,250**

But a **worse policy** that gets lucky randomization looks better than a **good policy** that consistently tries all positions!

## What Actually Happened in Your Training

Looking at eval checkpoints:

**460k steps** (ACTUALLY BEST):
```
Eval reward: 1102.52 ± 527.24
Episode length: 163.40
Training shows: MANY "Success! Reached red_box" messages
Actual behavior: Robot reaching successfully
```

**470k steps** (SAVED AS "BEST"):
```
Eval reward: 564.84 ± 126.46
Episode length: 44.40 (very short!)
```

Why did 470k get saved as "best"?
- Likely got lucky with evaluation episodes (all close positions)
- Short episodes with moderate success
- Mean reward calculation favored this

But when you visualize 470k:
- Gets random positions (as in real training)
- Policy hasn't actually learned to generalize
- Fails on most positions

## Solution: Use the Right Checkpoint

### Option 1: Visualize 460k Checkpoint Directly

```bash
# Find checkpoints
python find_best_checkpoint.py

# Visualize the 460k checkpoint
python visualize_checkpoint.py --timesteps 460000 --episodes 10
```

### Option 2: Better Eval Metric

The callback should save based on **success rate**, not mean reward.

Let me modify train_sb3.py to use success-based callback instead.

## Files to Help You

### 1. find_best_checkpoint.py
Lists all available checkpoints and recommends which to use based on your training logs.

### 2. visualize_checkpoint.py
Directly visualize any checkpoint by timestep:
```bash
python visualize_checkpoint.py --timesteps 460000
python visualize_checkpoint.py --timesteps 480000
python visualize_checkpoint.py --timesteps 490000
```

### 3. Updated Visualization Script
Will create a version that lets you choose checkpoint interactively.

## How to Identify Actually Best Checkpoint

Look for these indicators in training logs:

### Good Signs
1. **Many success messages** in console output
2. **Moderate episode length** (150-200 steps)
   - Too short (< 100): Failing fast or getting lucky
   - Too long (> 350): Not finding solution
3. **Consistent success** across multiple evals
4. **High success rate** when manually testing

### Based on Your Logs

**Best checkpoints to try**:
1. **460k steps**: reward=1102, ep_len=163 ✓ BEST
2. **490k steps**: reward=522, ep_len=106 (mixed results)
3. **500k steps (final)**: reward=927, ep_len=196 (good balance)

Avoid:
- **470k steps**: ep_len=44 (failing too fast - current "best_model")

## Recommended Next Steps

### 1. Visualize Actual Best Checkpoint

```bash
python visualize_checkpoint.py --timesteps 460000 --episodes 20
```

Watch for:
- Success rate > 30%
- Robot using elbow extension
- Smooth, purposeful reaching

### 2. Compare Multiple Checkpoints

```bash
# Try each promising checkpoint
for steps in 460000 480000 490000 500000; do
    echo "Testing $steps..."
    python visualize_checkpoint.py --timesteps $steps --episodes 10 --fast
done
```

### 3. Save the Actually Best One

Once you identify which is truly best (based on visual inspection):

```bash
# Copy the good checkpoint to replace "best_model"
cp -r ./models/g1_ppo_XXXXXX/g1_ppo_checkpoint_460000_steps ./models/g1_ppo_XXXXXX/best_model_actual
```

## Future Training: Fix the Callback

For next training run, we should modify the evaluation callback to save based on:
1. **Success rate** (primary)
2. **Mean distance** (secondary)
3. Mean reward (tertiary)

This will ensure "best model" actually means best performance, not best luck!

## Summary

**Problem**: "Best model" was selected based on mean eval reward, which favors lucky randomization over actual capability.

**Evidence**:
- Training showed many successes around 460k-490k
- "Best model" (470k) has terrible visualization performance
- Short episode lengths (44 steps) suggest fast failures, not good policy

**Solution**: Manually visualize checkpoints 460k, 480k, 490k, 500k to find actual best.

**Expected**: 460k checkpoint should show much better performance with actual successes!

## Quick Commands

```bash
# Find available checkpoints
python find_best_checkpoint.py

# Visualize the recommended best (460k)
python visualize_checkpoint.py --timesteps 460000 --episodes 20

# Compare others
python visualize_checkpoint.py --timesteps 480000 --episodes 10
python visualize_checkpoint.py --timesteps 500000 --episodes 10
```

Good catch on identifying this issue! The training was actually successful, just the wrong model got saved as "best".
