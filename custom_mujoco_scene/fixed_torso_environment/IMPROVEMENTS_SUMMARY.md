# RL Training Improvements - Breaking the 0.3m Local Minimum

## Summary of Changes

This document summarizes all the improvements made to help your G1 robot break through the 0.3m local minimum and successfully reach the target.

---

## 1. Reward Function Overhaul (`g1_rl_environment.py:550-639`)

### Key Changes:
- **Exponential Distance Reward**: `exp(-10 * distance)`
  - Provides strong gradient at ALL distances, not just far away
  - Gets exponentially better as robot approaches target

- **Massive Success Bonus**: `+1000` when distance < 0.05m
  - Dominates all other rewards to strongly incentivize reaching the goal
  - Clear signal that success is the ultimate objective

- **Comfort Zone Penalty**: `-50` when distance > 0.2m
  - **This is the key to breaking the 0.3m local minimum!**
  - Penalizes "safe" strategies that stay far from target
  - Forces robot to take risks and get closer

- **Velocity Reward**: `+10 * max(0, velocity_toward_target)`
  - Encourages active movement toward the target
  - Prevents static "waiting" behavior
  - Only rewards movement in the correct direction

- **Dense Approach Bonus**: `+100 * (previous_distance - current_distance)`
  - Immediate step-by-step feedback for getting closer
  - `-10` penalty for moving away (mild to allow exploration)
  - Helps with credit assignment

### Removed Issues:
- Old proximity bonuses that started at 0.2m (encouraged premature stopping)
- Complex multi-stage reward shaping that created local minima

---

## 2. Action Scale Increase (`g1_rl_environment.py:23`)

**Changed from**: `action_scale=0.25`
**Changed to**: `action_scale=0.5`

### Why This Matters:
- 0.25 was too conservative, limiting fine motor control
- 0.5 allows precise movements needed for final approach
- Doubled control authority enables reaching closer distances
- Updated in:
  - `g1_rl_environment.py` (default parameter)
  - `train_sb3_improved.py` (config)
  - `train_sb3.py` (default parameter)

---

## 3. Curriculum Learning (`g1_rl_environment.py:309-346`)

### Progressive Difficulty System:

| Episode Range | X Distance | Y Range | Difficulty |
|--------------|------------|---------|------------|
| 0-1000 | 0.3-0.5m | ±0.15m | **Easy** - Closer targets |
| 1000-3000 | 0.4-0.6m | ±0.18m | **Medium** - Intermediate |
| 3000+ | 0.35-0.65m | ±0.2m | **Full** - Complete range |

### Benefits:
- Robot learns easier reaching tasks first
- Builds up skills progressively
- Prevents early frustration and policy collapse
- Better sample efficiency

---

## 4. PPO Hyperparameter Tuning

### Updated Parameters:

| Parameter | Old Value | New Value | Purpose |
|-----------|-----------|-----------|---------|
| `learning_rate` | 5e-4 | **3e-4** | More stable learning |
| `n_steps` | 2048 | **4096** | Better advantage estimation |
| `batch_size` | 256 | **512** | More stable gradients |
| `ent_coef` | 0.0035 | **0.01** | Increased exploration |
| `clip_range` | 0.2 | **0.1** | Tighter clipping for fine control |
| `target_kl` | 0.02 | **0.01** | More conservative updates |

### Why These Changes:
- **Lower learning rate**: Prevents overshooting optimal policies
- **Larger rollouts**: Better estimates of advantage function
- **Bigger batches**: Reduces gradient variance
- **Higher entropy**: Encourages exploration near target
- **Tighter clipping**: Prevents large policy changes during fine-tuning
- **Lower target KL**: Stops learning if policy changes too much (stability)

---

## 5. Success-Based Reset (`g1_rl_environment.py:239-276`)

### Implementation:
After successful episodes, robot resets with position variations:
- X and Y position: `± 2cm` (normal distribution, σ=0.02)
- Z position (height): **No variation** (safety)

### Benefits:
- Improves policy generalization
- Prevents overfitting to exact starting position
- Builds robustness to initial state variations
- Real-world readiness

---

## 6. Dense Shaping Rewards (Already Included in Reward Function)

The approach bonus provides immediate feedback:
- `+100` per meter of progress toward target per step
- `-10` for moving away (allows exploration)
- Calculated every single step
- Strong immediate signal for learning

---

## Files Modified

1. **`g1_rl_environment.py`**
   - Reward function completely rewritten
   - Curriculum learning in `_randomize_object_positions()`
   - Success-based reset in `_reset_robot_pose()`
   - Action scale default increased
   - Episode tracking added

2. **`train_sb3.py`**
   - PPO hyperparameters updated
   - Function signature extended with new parameters
   - Model creation uses new configurable parameters
   - Action scale default increased

3. **`train_sb3_improved.py`**
   - Config dictionary updated with all new parameters
   - Documentation updated to reflect changes
   - Action scale increased

---

## Expected Results

With these improvements, you should see:

1. **Breaking through 0.3m barrier**
   - Comfort zone penalty forces robot to move closer
   - Velocity reward encourages active approach

2. **Faster convergence**
   - Curriculum learning starts with easier tasks
   - Better PPO parameters improve learning stability

3. **Higher success rate**
   - Massive success bonus provides clear objective
   - Exponential distance reward maintains strong gradient

4. **Smoother policies**
   - Larger batches and conservative updates
   - Tighter clipping prevents erratic behavior

---

## Training Recommendations

1. **Start fresh**: These are significant changes - start a new training run
2. **Monitor distance**: Watch if robot consistently gets below 0.25m
3. **Check velocity reward**: Should be positive, indicating movement toward target
4. **Episode length**: May need adjustment if robot reaches faster
5. **Curriculum stages**: Ensure you train through all 3 stages (>3000 episodes)

---

## Key Insight

**The 0.3m local minimum was caused by:**
- Reward structure that allowed "satisficing" (getting some reward without full success)
- Lack of penalty for staying far away
- Insufficient action authority for fine control
- No incentive for active movement toward target

**The solution:**
- Comfort zone penalty breaks the safe strategy
- Velocity reward encourages active approach
- Increased action scale enables precise control
- Exponential reward maintains gradient throughout approach

---

## Questions?

If training still plateaus:
1. Check if comfort penalty is activating (should see -50 in reward when >0.2m)
2. Verify velocity reward is positive when moving toward target
3. Ensure curriculum is progressing (check episode_count)
4. Monitor approach bonus - should be positive most steps

Good luck with training! These changes should help your robot push through to success! 🎯
