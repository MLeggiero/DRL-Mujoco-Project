# Smoothness Update - Eliminating Shakiness

## Problem

After anti-oscillation fixes, the robot was still exhibiting shaky, jerky movements instead of smooth reaching behavior.

## Root Causes Identified

1. **No penalty for action changes**: Robot could switch between extreme actions rapidly
2. **No incentive to hold position**: When close to target, robot kept moving
3. **Action scaling too high**: 0.5 scaling allowed fast, aggressive movements

## Solutions Applied

### 1. Smoothness Penalty (NEW)

**Location**: [g1_rl_environment.py:479-484](g1_rl_environment.py#L479-L484)

```python
# Track last action
self.last_action = None  # Added to __init__

# Smoothness penalty in reward function
if self.last_action is not None:
    action_diff = np.linalg.norm(action - self.last_action)
    smoothness_penalty = -0.1 * action_diff
```

**Effect**:
- Penalizes large changes between consecutive actions
- Encourages smooth, gradual movements
- L2 norm means all joints contribute to penalty
- Max penalty: -0.1 × 2.0 = -0.2 per step (if action flips completely)

### 2. Hold Position Reward (NEW)

**Location**: [g1_rl_environment.py:486-491](g1_rl_environment.py#L486-L491)

```python
if distance < 0.15:  # When within 15cm
    action_magnitude = np.linalg.norm(action)
    hold_reward = 0.5 * (1.0 - action_magnitude)
```

**Effect**:
- When close to target, rewards staying still
- Max reward: +0.5 for zero action (completely still)
- Gradually decreases as action magnitude increases
- Only active when within 15cm of target

### 3. Reduced Action Scaling

**Location**: [g1_rl_environment.py:337](g1_rl_environment.py#L337)

**Changed**:
```python
# OLD:
self.data.ctrl[actuator_id] = action[i] * 0.5

# NEW:
self.data.ctrl[actuator_id] = action[i] * 0.2
```

**Effect**:
- 60% reduction in action scaling (0.5 → 0.2)
- Slower, more controlled movements
- Reduces momentum and overshooting
- Makes smoothness penalties more effective

### 4. Increased Episode Length

**Location**: [g1_rl_environment.py:28](g1_rl_environment.py#L28)

**Changed**:
```python
# OLD:
self.max_episode_steps = 300

# NEW:
self.max_episode_steps = 500
```

**Effect**:
- 67% more time per episode
- Accommodates slower movements from 0.2 action scaling
- Robot can still reach target despite slower speed

## Complete Reward Structure

### All Components (in execution order)

1. **Distance Reward**: `-distance`
   - Range: -0.75 to -0.05
   - Primary signal to get closer

2. **Progress Reward**: `progress × 1.0` (if forward), `progress × 2.0` (if backward)
   - Rewards net forward progress
   - Punishes moving away (anti-oscillation)

3. **Proximity Bonuses**: Up to +37.0
   - 30cm: +2.0
   - 15cm: +5.0
   - 8cm: +10.0
   - 5cm (success): +20.0

4. **Smoothness Penalty**: `-0.1 × ||action - last_action||`
   - Range: 0.0 to -0.2
   - Discourages jerky movements

5. **Hold Position Reward**: `0.5 × (1.0 - ||action||)` (if distance < 15cm)
   - Range: 0.0 to +0.5
   - Rewards staying still when close

6. **Time Penalty**: -0.001
   - Encourages efficiency

### Total Reward Range

**Far from target (>30cm)**: -0.75 to -0.3 per step
**Approaching (15-30cm)**: -0.3 to +0.5 per step
**Close (<15cm)**: -0.15 to +2.0 per step (with hold bonus)
**Touching (<5cm)**: +15 to +25 per step

## Expected Behavior Changes

### Before (Shaky)
- ❌ Rapid action changes (jerky movements)
- ❌ Oscillating when close to target
- ❌ High velocities, overshooting
- ❌ Never holding position

### After (Smooth)
- ✅ Gradual action changes (smooth movements)
- ✅ Settling when close to target
- ✅ Controlled approach, minimal overshoot
- ✅ Holding position near target

## Training Implications

### Slower Initial Learning

With 0.2 action scaling, early exploration will be slower:
- **0-40k steps**: Learning basic arm control
- **40-70k steps**: Discovering forward reaching
- **70-100k steps**: Fine-tuning smooth approach

### Better Final Performance

The smoothness constraints will lead to:
- More stable policies
- Better generalization
- Human-like movements
- Higher success rate

## Comparison: Gaming vs Oscillation vs Shakiness

| Behavior | Cause | Fix Applied |
|----------|-------|-------------|
| **Gaming** | Progress reward too high (10.0) | Reduced to 1.0 |
| **Oscillation** | Equal reward for forward/back | Asymmetric penalty (2.0x back) |
| **Shakiness** | No action smoothness constraint | Smoothness penalty + hold reward |

All three problems required different solutions!

## Testing the Changes

### Quick Test
```bash
python train_sb3_improved.py
```

### Monitor Smoothness
When visualizing, look for:
1. ✅ Gradual arm extension (not jerky)
2. ✅ Smooth approach to target
3. ✅ Settling behavior when close
4. ✅ Minimal oscillation near target

```bash
./scripts/visualize.sh
```

### Expected Training Metrics

**Early (0-40k)**:
- Reward: -0.5 to -0.3
- Distance: 0.75m → 0.4m
- Behavior: Slow exploration, learning arm control

**Mid (40-70k)**:
- Reward: -0.3 to 0.0
- Distance: 0.4m → 0.2m
- Behavior: Purposeful reaching, getting closer

**Late (70-100k)**:
- Reward: 0.0 to +1.0
- Distance: 0.2m → 0.05-0.1m
- Behavior: Smooth approach, some touches, holding position

## Files Modified

1. **g1_rl_environment.py**
   - Line 28: Episode length 300 → 500
   - Line 41: Added `last_action` tracking
   - Line 173-174: Reset `last_action` in reset()
   - Line 307: Pass action to reward function
   - Line 337: Action scaling 0.5 → 0.2
   - Lines 449-503: Complete reward rewrite with smoothness

2. **train_sb3_improved.py**
   - Lines 53-72: Updated configuration description

## Summary

**Problem**: Shaky, jerky movements
**Causes**: No action smoothness, no hold incentive, action scaling too high
**Fixes**:
- Smoothness penalty: -0.1 × action changes
- Hold reward: +0.5 when close and still
- Action scaling: 0.5 → 0.2 (60% reduction)
- Episode length: 300 → 500 steps

**Expected**: Smooth, controlled reaching with position holding near target
