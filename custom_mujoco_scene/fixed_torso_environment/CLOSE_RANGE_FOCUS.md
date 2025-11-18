# Close-Range Focus Update - Rewarding Only Very Close Proximity

## User Request

"Change the rewards to be mostly rewarding when the hand gets very close to the red box"

## Problem with Previous Reward Structure

The robot was getting too much reward at intermediate distances:

**Previous rewards at different distances**:
```
50cm: -0.25 (distance²)
30cm: -0.09 + 10 = ~+10
15cm: -0.02 + 40 = ~+40   ← Too much reward here!
10cm: -0.01 + 40 = ~+40   ← Same reward as 15cm
8cm:  -0.006 + 100 = ~+100
5cm:  -0.0025 + 250 = ~+250
```

**Issue**: Robot gets +40 at 15cm with minimal effort, then only +210 more to actually touch. Not motivating enough.

## New Reward Structure: Extreme Close-Range Focus

### Proximity Bonuses - Heavily Weighted to <10cm

**Old structure** (too generous early):
```
15cm: +40 cumulative
8cm:  +100 cumulative
5cm:  +250 cumulative
```

**NEW structure** (almost nothing until very close):
```
15cm: +5 cumulative   (tiny bonus - robot learns this isn't enough)
10cm: +25 cumulative  (small bonus - getting warmer)
8cm:  +75 cumulative  (decent bonus - closer!)
6cm:  +175 cumulative (huge jump - almost there!)
5cm:  +475 cumulative (MASSIVE - success!)
```

### Reward Breakdown by Distance

| Distance | Dist Reward | Proximity | Total/step | % of Max |
|----------|-------------|-----------|------------|----------|
| 50cm | -5.0 | 0 | -5 | 0% |
| 30cm | -3.0 | 0 | -3 | 0% |
| 15cm | -1.5 | +5 | +3.5 | 0.7% |
| 10cm | -1.0 | +25 | +24 | 5% |
| 8cm | -0.8 | +75 | +74 | 16% |
| 6cm | -0.6 | +175 | +174 | 37% |
| **5cm** | **-0.5** | **+475** | **+474** | **100%** |

### Key Changes

1. **Distance Reward**: Changed from `-d²` to `-10*d`
   - Linear instead of quadratic
   - Simpler gradient
   - Doesn't dominate at far distances

2. **Progress Reward**: Only active when distance < 20cm
   ```python
   # OLD: Always active, 5x multiplier
   if progress > 0:
       progress_reward = progress * 5.0

   # NEW: Only when close, 20x multiplier
   if distance < 0.2 and progress > 0:
       progress_reward = progress * 20.0  # But only if already close!
   ```

3. **Proximity Thresholds**: Added 10cm and 6cm, removed 30cm
   ```python
   # Removed: 30cm threshold (was giving too much too early)
   if distance < 0.15:  proximity_bonus += 5    # Was +10, now +5
   if distance < 0.10:  proximity_bonus += 20   # NEW threshold
   if distance < 0.08:  proximity_bonus += 50   # Was +60, now +50
   if distance < 0.06:  proximity_bonus += 100  # NEW threshold (huge jump!)
   if distance < 0.05:  proximity_bonus += 300  # Was +150, now +300
   ```

## Why This Forces Full Extension

### The Economics

**Strategy 1: Shoulder only (gets to 15cm)**
- Reward per step: ~+3.5
- Total episode: 400 steps × +3.5 = **1,400**

**Strategy 2: Shoulder only (gets to 10cm with luck)**
- Reward per step: ~+24
- Total episode: 400 steps × +24 = **9,600**

**Strategy 3: Shoulder + elbow (gets to 6cm)**
- Reward per step: ~+174
- Total episode: 300 steps × +174 = **52,000**

**Strategy 4: Full extension (touches at 5cm)**
- Reward per step: ~+474
- Total episode: 300 steps × +474 = **142,000**

**10x difference** between "good enough" (15cm) and success (5cm)!

### Reward Gradient

The gradient now creates a "reward cliff":
```
15cm → 10cm: +20.5 more per step (6x increase)
10cm → 8cm:  +50 more per step (3x increase)
8cm → 6cm:   +100 more per step (2.4x increase)
6cm → 5cm:   +300 more per step (2.7x increase)
```

Each centimeter closer in the final range gives exponentially more reward.

## Expected Behavior

### Phase 1: Discovery (0-30k steps)
- Learning basic arm control
- Getting to ~30cm
- Rewards: -5 to 0
- **Not getting any bonuses yet**

### Phase 2: Initial Approach (30-50k steps)
- Shoulder rotation working
- Getting to ~15cm
- Rewards: 0 to +500 per episode
- **Realizes: "This isn't much reward, need to go closer"**

### Phase 3: Exploration (50-70k steps)
- Trying different movements
- **Discovering elbow extension**
- Occasionally hitting 10cm, sometimes 8cm
- Rewards: +500 to +5,000 per episode
- **Realizes: "Big rewards when I get really close!"**

### Phase 4: Optimization (70-100k steps)
- Coordinating shoulder + elbow
- Consistently reaching 6-8cm
- Some successes (<5cm)
- Rewards: +5,000 to +50,000 per episode
- **Learning: "Full extension = massive rewards"**

## Comparison Chart

### Old vs New Reward Distribution

```
Old rewards:
15cm ████████████████ +40 (40% of max at success)
10cm ████████████████ +40 (same!)
8cm  ████████████████████████████ +100 (85% of max)
5cm  ██████████████████████████████ +250 (100%)

New rewards:
15cm █ +5 (1% of max at success)
10cm ███ +25 (5% of max)
8cm  ████████ +75 (16% of max)
6cm  ████████████████████ +175 (37% of max)
5cm  ██████████████████████████████ +475 (100%)
```

**Key**: Robot gets almost nothing until it's VERY close.

## Training Implications

### Slower Initial Learning
- Robot won't see big rewards early
- First 30-50k steps will show modest progress
- This is OK - we want the robot to learn the full solution

### Explosive Mid-Late Learning
- Once robot discovers getting close (50-70k)
- Rewards will jump dramatically
- Should see rapid improvement from 15cm → 5cm

### Success Indicators

Watch for these milestones in training:

**30k steps**:
- Distance: 0.5m → 0.2m
- Reward: ~-2 per step
- "Learning basic approach"

**50k steps**:
- Distance: 0.2m → 0.15m
- Reward: ~+3 per step
- "Hitting first threshold"

**70k steps**:
- Distance: 0.15m → 0.08m
- Reward: ~+50 per step
- "Elbow discovery!"

**90k steps**:
- Distance: 0.08m → 0.06m
- Reward: ~+150 per step
- "Approaching success"

**100k steps**:
- Distance: ~0.05-0.06m
- Reward: ~+300 per step
- "First successes"

## Files Modified

1. **g1_rl_environment.py** (Lines 453-494)
   - Changed distance reward: `-d²` → `-10*d`
   - Progress reward only when `distance < 0.2`
   - New proximity structure:
     - 15cm: +5 (was +40)
     - Added 10cm: +20
     - 8cm: +50 (was +100)
     - Added 6cm: +100
     - 5cm: +300 (was +250)

2. **train_sb3_improved.py** (Lines 64-74)
   - Updated reward description

## Summary

**Changed**: Reward structure to give almost nothing until hand is very close (<10cm)

**Effect**:
- 15cm: Only +5 (was +40) - robot learns this isn't enough
- 10cm: +25 total - starting to see rewards
- 6cm: +175 total - huge jump, very motivating
- 5cm: +475 total - massive success reward

**Expected**: Robot forced to learn full arm extension (including elbow) to get meaningful rewards.

**Retrain**:
```bash
python train_sb3_improved.py
```

Watch for rewards to stay low until ~50k steps, then explode as robot learns to get very close!
