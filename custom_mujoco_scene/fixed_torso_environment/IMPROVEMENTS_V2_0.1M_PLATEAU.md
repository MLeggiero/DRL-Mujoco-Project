# V2 Improvements - Breaking the 0.1m Plateau

## Situation Analysis

Based on your training graph screenshot:

### Current Performance:
- ✅ **Successfully broke through 0.3m barrier** (down to ~0.1m)
- ✅ **67% improvement** in distance
- ✅ **Rewards improved** from -37k to -7k
- ❌ **New plateau at 0.1-0.12m** (need to reach 0.05m)

### Root Cause:
The original comfort zone penalty (`-50` at `>0.2m`) is no longer active at 0.1m, so the robot found a new "safe zone" where it gets decent rewards without the risk of trying to get closer.

---

## New Improvements (V2)

### 1. Tiered Comfort Zone Penalties ⭐ **Most Important**

**Old System:**
```python
if distance > 0.2:
    comfort_penalty = -50.0
```

**New Tiered System:**
```python
if distance > 0.15:
    comfort_penalty = -100.0  # Strong penalty (breaks 0.3m)
elif distance > 0.08:
    comfort_penalty = -80.0   # NEW: Breaks 0.1m plateau!
elif distance > 0.06:
    comfort_penalty = -40.0   # Final push to goal
```

**Why This Works:**
- At 0.1m: Robot now gets `-80` penalty → forced to push closer
- Creates a "ladder" of penalties that push robot all the way to goal
- No comfortable stopping points between 0.15m and 0.06m

---

### 2. Increased Success Bonus

**Changed:** `1000` → `5000` (5x increase!)

**Reasoning:**
- Makes reaching the goal absolutely irresistible
- Success bonus now dominates ALL other rewards combined
- At 0.1m, robot sees: "I could get -80/step now, OR +5000 if I reach 5cm more"

---

### 3. Steeper Exponential Distance Reward

**Changed:** `exp(-10 * distance)` → `exp(-15 * distance)`

**Impact at Different Distances:**

| Distance | Old Reward | New Reward | Improvement |
|----------|------------|------------|-------------|
| 0.20m | 0.135 | 0.050 | Weaker (good - penalty zone) |
| 0.10m | 0.368 | 0.223 | Moderate gradient |
| 0.08m | 0.449 | 0.301 | Strong gradient |
| 0.05m | 0.606 | 0.472 | Very strong! |

The steeper curve provides stronger gradient in the 0.05-0.10m range.

---

### 4. Proximity Bonuses (NEW!)

Added milestone rewards for getting closer:

```python
if distance < 0.15:
    proximity_bonus += 50.0   # Entered close range
if distance < 0.10:
    proximity_bonus += 100.0  # Breaking the barrier!
if distance < 0.08:
    proximity_bonus += 200.0  # Very close
if distance < 0.06:
    proximity_bonus += 500.0  # Almost there!
```

**These are cumulative!** At 0.06m, robot gets `50+100+200+500 = +850` bonus.

---

### 5. Doubled Approach Bonus

**Changed:** `100 * Δdistance` → `200 * Δdistance`

**Impact:**
- Every 1cm closer = +2.0 reward (was +1.0)
- Moving from 0.10m to 0.09m = instant +2.0 reward
- Stronger immediate feedback for making progress

**Penalty for moving away:** Also doubled from `-10` to `-20`

---

## Expected Reward Comparison

### At 0.10m distance (current plateau):

**Old Rewards:**
- Exponential: `exp(-10*0.1)` = +0.37
- Success: 0
- Comfort penalty: 0 (not active!)
- Proximity: 0
- **Total static reward: ~+0.37** ✗ Comfortable!

**New Rewards:**
- Exponential: `exp(-15*0.1)` = +0.22
- Success: 0
- Comfort penalty: **-80** ✓ Uncomfortable!
- Proximity: +150 (crossed 0.15m and 0.10m thresholds)
- **Total static reward: ~+70.22** with strong pressure to move

---

### At 0.06m distance (near goal):

**New Rewards:**
- Exponential: `exp(-15*0.06)` = +0.41
- Success: 0 (not yet)
- Comfort penalty: **-40** (still pushing!)
- Proximity: **+850** (all bonuses!)
- **Total static reward: ~+810** ✓ Much better than 0.1m!

---

### At 0.04m distance (SUCCESS!):

**New Rewards:**
- Exponential: `exp(-15*0.04)` = +0.55
- Success: **+5000** 🎯
- Comfort penalty: 0
- Proximity: +850
- **Total static reward: ~+5850** 🎉

---

## Complete Reward Structure Summary

### Distance-Based Components:
1. **Exponential**: `exp(-15*d)` - Continuous gradient
2. **Tiered Penalties**: Push through plateaus
3. **Proximity Bonuses**: Milestone rewards
4. **Success Bonus**: Massive reward at goal

### Movement-Based Components:
5. **Velocity Reward**: `+10*v_toward` - Encourage movement
6. **Approach Bonus**: `+200*Δd` - Immediate progress feedback

### Regularization (penalties):
7. **Joint Velocity**: `-0.005*Σ(qvel²)` - Smooth movements
8. **Action Magnitude**: `-0.005*Σ(action²)` - Energy efficiency
9. **Smoothness**: `-weight*Σ(Δaction²)` - No jerking

---

## Why This Will Work

### The 0.1m Plateau Problem:
**Before:** At 0.1m, robot got ~+0.37 reward with no penalty → comfortable stop
**After:** At 0.1m, robot gets -80 penalty + must see +850 available at 0.06m → forced to move

### The Gradient Path:
The robot now sees a clear "reward ladder":
- 0.15m → 0.10m: Escape -100 penalty, get +150 bonus
- 0.10m → 0.08m: Escape -80 penalty, get +200 more bonus
- 0.08m → 0.06m: Escape -40 penalty, get +500 more bonus
- 0.06m → 0.05m: Escape -40 penalty, get **+5000** success!

**No comfortable stopping points!**

---

## Training Recommendations

### 1. Continue Current Training
You can continue from your current checkpoint! The new rewards will immediately apply.

### 2. Watch for These Signs:
- Distance dropping below 0.08m consistently
- Comfort penalty showing -80 to -100 in logs
- Proximity bonuses appearing (+150, +350, +850)
- Success messages: "🎯 SUCCESS!"

### 3. Expected Timeline:
- Next 100k steps: Should break below 0.08m
- Next 300k steps: Should start seeing successes at 0.05m
- By 1.5M total steps: Regular successes expected

### 4. If Still Stuck at 0.08m:
Consider increasing action scale to 0.6-0.7 for even finer control.

---

## Comparison Table: All Changes

| Component | Original | V1 (0.3m fix) | V2 (0.1m fix) |
|-----------|----------|---------------|---------------|
| Exponential steepness | -8 | -10 | **-15** |
| Success bonus | 2000 | 1000 | **5000** |
| Comfort penalty | Single @0.3m | Single @0.2m | **Tiered (3 levels)** |
| Proximity bonuses | Multiple | None | **4 milestones** |
| Approach multiplier | 50 | 100 | **200** |
| Action scale | 0.25 | 0.5 | 0.5 |

---

## Files Modified

1. **`g1_rl_environment.py`** - Reward function enhanced (lines 606-686)
2. **`train_sb3_improved.py`** - Documentation updated (lines 110-119)

---

## Bottom Line

Your robot WAS finding it comfortable at 0.1m with small positive rewards and no penalties.

Now at 0.1m it gets **-80 penalty** and can clearly see **+850 bonus** waiting at 0.06m, plus **+5000** at the goal.

**The comfort zone is broken. The path forward is clear. Time to reach! 🎯**

---

## Quick Start

To use these improvements:
```bash
cd /home/mleggiero/rl_training/DRL-Mujoco-Project/custom_mujoco_scene/fixed_torso_environment
python train_sb3_improved.py --scene ../unitree_g1/g1_table_box_scene.xml
```

Or continue your current training - the new rewards apply immediately!
