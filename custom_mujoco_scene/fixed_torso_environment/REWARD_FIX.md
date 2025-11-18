# Reward Function Fix - Nov 17, 2025

## Problem Identified

After training with the improved environment (50k timesteps), the robot showed:
- **High rewards**: -101 → +957 (appeared to be learning)
- **Poor actual performance**: Minimum distance only 0.17-0.25m (not reaching)
- **No successes**: 0/5 episodes succeeded, all ended at 301 steps
- **Incoherent movement**: Robot moving erratically, not purposefully

### Root Cause: Reward Gaming

The robot was exploiting the reward function by **oscillating** near the target rather than actually reaching it.

**Problem component**: `progress_reward = progress * 10.0`

With a 10x multiplier, the robot discovered it could get continuous high rewards by:
1. Moving toward target → get +reward
2. Moving away from target → lose some reward
3. Moving toward again → get +reward again
4. Repeat oscillation → accumulate massive rewards without reaching

This is classic **reward gaming** behavior.

## Fix Applied

### Changes to Reward Function

**File**: [g1_rl_environment.py:448-485](g1_rl_environment.py#L448-L485)

#### 1. Reduced Progress Reward Multiplier
```python
# OLD (caused gaming):
progress_reward = progress * 10.0

# NEW (prevents gaming):
if progress > 0:
    progress_reward = progress * 2.0  # 5x reduction
else:
    progress_reward = progress * 0.5  # Small penalty for moving away
```

**Reasoning**:
- 2.0x multiplier still rewards progress, but not enough to make oscillation profitable
- Asymmetric penalty (0.5x) for moving away discourages back-and-forth motion
- Robot must now make net forward progress to maximize reward

#### 2. Enhanced Proximity Bonuses
```python
# OLD:
if distance < 0.3:   proximity_bonus += 1.0
if distance < 0.15:  proximity_bonus += 3.0
if distance < 0.05:  proximity_bonus += 10.0

# NEW:
if distance < 0.3:   proximity_bonus += 2.0   # Increased
if distance < 0.15:  proximity_bonus += 5.0   # Increased
if distance < 0.08:  proximity_bonus += 10.0  # New threshold
if distance < 0.05:  proximity_bonus += 20.0  # Increased
```

**Reasoning**:
- Stronger gradient toward actual reaching
- New 8cm threshold provides intermediate feedback
- Success bonus doubled (20.0) to heavily emphasize completion

#### 3. Core Components Unchanged
```python
distance_reward = -distance  # Still primary signal
time_penalty = -0.001        # Still encourages efficiency
```

## New Reward Structure Summary

### Reward Components (in order of importance):

1. **Distance Reward**: -distance
   - Primary signal: being closer is better
   - Range: -0.75 (far) to -0.05 (touching)

2. **Proximity Bonuses**: Up to +37.0 for success
   - 30cm: +2.0
   - 15cm: +5.0 (cumulative: +7.0)
   - 8cm: +10.0 (cumulative: +17.0)
   - 5cm (success): +20.0 (cumulative: +37.0)

3. **Progress Reward**: Up to +2.0 per step
   - Forward progress: progress × 2.0
   - Backward movement: progress × 0.5 (penalty)

4. **Time Penalty**: -0.001 per step
   - Encourages efficiency without overwhelming signal

### Expected Reward Ranges

**Random policy**: -0.75 to -0.5 per step (just distance)
**Learning policy**: -0.3 to 0.0 per step (getting closer + bonuses)
**Good policy**: 0.0 to +2.0 per step (close + progress)
**Successful episode**: +20 to +40 total (reaching target)

## Training Implications

### Training Duration Adjustment

With the reduced progress reward, the robot may need **more exploration time** to discover reaching behavior:

**Recommended**: Increase from 50k to **100k timesteps**

Rationale:
- Lower progress reward = slower initial learning
- But prevents gaming, so final performance will be better
- 100k gives enough time to discover effective policy

### Expected Training Progression

**0-20k steps**: Random exploration, rewards -0.5 to -0.3
- Learning basic arm movements
- Distance improving from 0.75m to 0.5m

**20k-50k steps**: Purposeful approach, rewards -0.3 to 0.0
- Discovering forward-reaching motions
- Distance improving from 0.5m to 0.2m
- Getting proximity bonuses

**50k-100k steps**: Fine-tuning reach, rewards 0.0 to +2.0
- Accurate reaching, some successes
- Distance improving from 0.2m to 0.05-0.1m
- Regular proximity bonuses at all levels

### Success Criteria

After 100k timesteps:
- **Minimum acceptable**: Mean distance < 0.15m, rewards > -0.1
- **Good performance**: Mean distance < 0.10m, rewards > +0.5
- **Excellent performance**: Success rate > 20%, mean distance < 0.08m

## How to Retrain

### Quick Retrain (Recommended)
```bash
# Uses new reward function, 100k timesteps
python train_sb3_improved.py
```

### Monitor Progress
```bash
# In another terminal
tensorboard --logdir ./logs
```

### Visualize Checkpoints
```bash
# Check progress at 50k
./scripts/visualize.sh

# Check final policy at 100k
./scripts/visualize.sh
```

## What to Watch For

### Good Signs
- Reward steadily increasing (less negative)
- Distance to target decreasing over time
- Arm reaching forward purposefully in visualization
- Occasional success messages in training output

### Bad Signs (If They Occur)
- Rewards oscillating wildly → Learning rate too high
- Distance not decreasing after 30k steps → Action scaling too low
- Robot completely still → Need to check actuator setup
- Rewards increasing but distance not decreasing → Still gaming somehow

## Verification After Training

After training completes, run visualization and check:

```bash
./scripts/visualize.sh
```

**Expected behavior**:
1. Arm reaches forward toward red box
2. Smooth, purposeful motion (not jerky oscillation)
3. Gets within 5-10cm of target
4. Some episodes should succeed (touch the box)

**NOT expected**:
1. ❌ Oscillating/vibrating in place
2. ❌ Moving backward after getting close
3. ❌ Jerky, incoherent movements
4. ❌ Staying far from target (>20cm)

## Files Changed

1. [g1_rl_environment.py](g1_rl_environment.py)
   - Lines 448-485: Reward function restructured

2. [train_sb3_improved.py](train_sb3_improved.py) (to be updated)
   - Line 14: Increase to 100k timesteps

## Summary

**Problem**: Robot gaming reward function by oscillating
**Cause**: Progress reward multiplier too high (10.0)
**Fix**: Reduced to 2.0 with asymmetric penalty for backward motion
**Impact**: Robot must now make net forward progress, not oscillate
**Training**: Increase to 100k timesteps to compensate for reduced reward signal
**Expected**: Actual reaching behavior, not reward gaming
