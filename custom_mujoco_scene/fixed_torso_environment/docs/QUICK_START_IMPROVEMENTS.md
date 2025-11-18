# Quick Start: Improving Your Training Results

Your first training run was **successful but suboptimal**. Here's how to get much better results in 3 simple steps.

## Your Current Results

```
Best reward: -719 (at 30k steps)
Final reward: -863 (at 100k steps)
Success rate: 0%
Distance: ~0.7-0.9m from target
```

## Expected Results After Improvements

```
Best reward: -50 to +5
Success rate: 20-50%
Distance: ~0.1-0.3m from target
```

---

## 3-Step Improvement Process

### Step 1: Test Current Model (2 minutes)

See what your current model actually achieves:

```bash
python train_sb3.py --test ./models/g1_ppo_20251117_133846/best_model/best_model
```

This will run 5 test episodes and show:
- Average reward
- Average distance to target
- Success rate

**Expected**: Distance ~0.7-0.9m, 0% success

---

### Step 2: Apply Quick Fixes (1 minute)

**Option A: Automatic (recommended)**
```bash
./apply_quick_fixes.sh
```

**Option B: Manual**

Edit `g1_rl_environment.py`:

1. Line 27: Change episode length
   ```python
   self.max_episode_steps = 300  # Was 1000
   ```

2. Line 384: Reduce time penalty
   ```python
   time_penalty = -0.001  # Was -0.01
   ```

3. Lines 372-377: Increase proximity rewards
   ```python
   if distance < 0.3:
       proximity_bonus += 1.0  # Was 0.5
   if distance < 0.15:
       proximity_bonus += 3.0  # Was 1.0
   if distance < self.success_distance:
       proximity_bonus += 10.0  # Was 5.0
   ```

---

### Step 3: Retrain (30-60 minutes)

**Option A: Use improved config (recommended)**
```bash
python train_sb3_improved.py
```

This trains for 300k steps with optimized hyperparameters:
- Higher learning rate (5e-4)
- More frequent updates (1024 steps)
- Larger batches (128)
- More PPO epochs (20)

**Option B: Use standard config**
```bash
python train_sb3.py --timesteps 300000
```

**Monitor progress:**
```bash
# In another terminal
tensorboard --logdir ./logs
```

Open http://localhost:6006 to watch training in real-time.

---

## What Each Fix Does

### Fix 1: Episode Length (1000 → 300)
**Problem**: 1000 steps is way too long. Robot spends most of the time wandering after giving up.

**Impact**:
- Faster learning (sees episode endings more often)
- Less time penalty accumulation
- Forces agent to learn efficiency

**Expected improvement**: ⭐⭐⭐ Critical

---

### Fix 2: Time Penalty (-0.01 → -0.001)
**Problem**: Time penalty was dominating the reward signal.
- Old: -0.01 × 1000 steps = -10 per episode
- Distance reward: -0.7 (for 0.7m away)
- Time penalty was 14x larger than distance signal!

**Impact**:
- Distance reward becomes primary signal
- Agent focuses on reaching, not just surviving

**Expected improvement**: ⭐⭐⭐ Critical

---

### Fix 3: Proximity Rewards (doubled)
**Problem**: Rewards for getting close were too small.

**Old rewards**:
- Within 30cm: +0.5
- Within 15cm: +1.0
- Success (<5cm): +5.0

**New rewards**:
- Within 30cm: +1.0
- Within 15cm: +3.0
- Success (<5cm): +10.0

**Impact**: Stronger incentive to get close

**Expected improvement**: ⭐⭐ Important

---

### Fix 4: More Training (100k → 300k)
**Problem**: 100k steps not enough for complex manipulation.

**Impact**: Agent has 3x more experience to learn from

**Expected improvement**: ⭐⭐ Important

---

### Fix 5: Better Hyperparameters
**Problem**: Default SB3 hyperparameters aren't tuned for this task.

**Changes** (in `train_sb3_improved.py`):
- `learning_rate`: 3e-4 → 5e-4 (faster learning)
- `n_steps`: 2048 → 1024 (more frequent updates)
- `batch_size`: 64 → 128 (more stable)
- `n_epochs`: 10 → 20 (learn more from each batch)

**Expected improvement**: ⭐ Moderate

---

## Monitoring Your Training

### In Terminal
Watch for:
```
eval/mean_reward: Should increase over time
eval/mean_ep_length: Should decrease (episodes ending earlier)
```

### In TensorBoard
Key metrics:
- `rollout/ep_rew_mean`: Trending upward?
- `train/explained_variance`: Should be > 0.5
- `train/value_loss`: Should be decreasing

### Success Indicators
- Episodes ending before 300 steps (robot reached target!)
- Mean reward climbing toward 0 or positive
- "Success!" messages in terminal

---

## What to Expect

### After 50k steps (15-20 min)
- Reward: -300 to -200
- Distance: ~0.4-0.6m
- Agent moving toward target more directly

### After 150k steps (45 min)
- Reward: -100 to -50
- Distance: ~0.2-0.3m
- Occasional successes (5-10%)

### After 300k steps (90 min)
- Reward: -50 to +5
- Distance: ~0.1-0.2m
- Consistent successes (20-50%)

---

## Troubleshooting

### Still not learning?
1. Check that quick fixes were applied: `grep "max_episode_steps = 300" g1_rl_environment.py`
2. Make sure you're using the improved config: `python train_sb3_improved.py`
3. Let it train longer (500k steps)

### Training is slow?
- This is normal on CPU (170 steps/sec)
- 300k steps takes ~30-60 minutes
- Consider reducing to 100k for quick tests

### Reward still very negative?
- Check TensorBoard - is it trending upward?
- If flat, try even smaller time penalty (-0.0001)
- Or increase proximity rewards more

---

## Full Workflow

```bash
# 1. Test baseline
python train_sb3.py --test ./models/g1_ppo_20251117_133846/best_model/best_model

# 2. Apply fixes
./apply_quick_fixes.sh

# 3. Verify fixes
python test_environment.py

# 4. Retrain with improvements
python train_sb3_improved.py

# 5. Monitor (in another terminal)
tensorboard --logdir ./logs

# 6. Test improved model
python train_sb3.py --test ./models/g1_ppo_*/best_model/best_model

# 7. Compare results
```

---

## Expected Results Comparison

| Metric | Before | After Quick Fixes | After Full Training |
|--------|--------|-------------------|---------------------|
| Best Reward | -719 | -200 to -100 | -50 to +5 |
| Episode Length | 1000 steps | 300-600 steps | 100-300 steps |
| Distance | 0.7-0.9m | 0.3-0.5m | 0.1-0.2m |
| Success Rate | 0% | 5-15% | 20-50% |
| Training Time | 100k steps (10 min) | 300k steps (30-60 min) | 300k steps (30-60 min) |

---

## Bottom Line

The three critical changes are:

1. **Episode length**: 1000 → 300 steps
2. **Time penalty**: -0.01 → -0.001
3. **More training**: 100k → 300k steps

These alone should give you **5-10x better performance**.

Just run:
```bash
./apply_quick_fixes.sh
python train_sb3_improved.py
```

Come back in 30-60 minutes and you'll have a much better model!

---

## Need More Help?

See detailed explanations in:
- [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) - Full improvement guide
- [README_SB3.md](README_SB3.md) - SB3 usage documentation
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Why we switched to SB3
