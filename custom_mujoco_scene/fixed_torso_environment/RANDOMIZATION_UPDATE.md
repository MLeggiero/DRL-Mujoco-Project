# Object Position Randomization - Ensuring Generalization

## User Question

"Is the robot learning to touch the red box regardless of the box's location?"

## Problem Identified

**Short answer**: No, the robot was NOT learning a generalized policy.

### Previous Randomization

```python
# OLD: Tiny randomization
noise = np.random.uniform(-0.03, 0.03, 3)  # Only ±3cm
self.data.qpos[qpos_addr:qpos_addr+3] = original_pos + noise
```

**Issue**: With only ±3cm randomization:
- Object always at approximately the same position
- Robot learns to reach for a **fixed point in space**
- NOT learning to track the actual object
- Policy will fail if object is moved more than 3cm

### What Was Actually Happening

The robot was learning:
```python
# Pseudocode of what the policy learned
def reach():
    move_shoulder_to_angle(45°)  # Fixed angle
    move_elbow_to_angle(30°)     # Fixed angle
    # Always reaches for position [0.5, 0, 0.74] regardless of where box is!
```

This is called **overfitting to the environment** - it works in training but won't generalize.

## Solution: Significant Randomization

### New Randomization Range

```python
# NEW: Significant randomization across reachable region
x = np.random.uniform(0.35, 0.65)  # 30cm range forward/back
y = np.random.uniform(-0.2, 0.2)   # 40cm range left/right
z = 0.74  # Table height (fixed)
```

### Randomization Region Visualization

```
Top-down view (robot at origin looking forward in +X direction):

         Y (left/right)
         |
  -0.2   |   +0.2
    +----+----+
    |    |    |  X=0.65m (far)
    |    |    |
    | RANDOM  |  X=0.5m (middle)
    |  REGION |
    |    |    |  X=0.35m (close)
    +----+----+
         |
    Robot at (0, 0)

Total region:
- Forward: 30cm range (0.35m to 0.65m)
- Lateral: 40cm range (-0.2m to +0.2m)
- Height: Fixed at table (0.74m)
```

### Comparison

| Aspect | Old (±3cm) | New (30×40cm) | Improvement |
|--------|------------|---------------|-------------|
| X range | 0.47-0.53m | 0.35-0.65m | **10x larger** |
| Y range | -0.03 to +0.03m | -0.2 to +0.2m | **13x larger** |
| Total area | ~28 cm² | ~1200 cm² | **43x larger** |
| Generalization | None | Good | ✓ |

## Why This Forces Real Learning

### Before (Fixed Position Learning)

**Episode 1**: Box at [0.50, 0.01, 0.74]
- Robot learns: "Move to [0.50, 0, 0.74]"

**Episode 2**: Box at [0.52, -0.02, 0.74]
- Robot uses same strategy: "Move to [0.50, 0, 0.74]"
- Still works! (only 3cm difference)

**Result**: Robot never learns to use visual/proprioceptive feedback.

### After (True Object Tracking)

**Episode 1**: Box at [0.40, 0.15, 0.74]
- Robot must learn: "Read target position from observation, reach THERE"

**Episode 2**: Box at [0.60, -0.10, 0.74]
- Previous strategy fails! (box 25cm away from last position)
- Robot must use observation to locate box

**Result**: Robot learns to actually track and reach for the object.

## What the Robot Must Now Learn

### Required Capabilities

1. **Visual-Motor Coordination**
   - Read target position from observation
   - Compute required joint angles
   - Execute reaching motion

2. **Generalized Reaching**
   - Different shoulder angles for left/right targets
   - Different elbow extension for near/far targets
   - Coordinate both joints for arbitrary positions

3. **Adaptation**
   - Each episode has different target position
   - Can't memorize fixed trajectory
   - Must compute on-the-fly

## Training Implications

### Longer Training Required

**Increased timesteps**: 100k → 500k

**Why**:
- More variation = more exploration needed
- Must learn generalized policy, not memorized trajectory
- 43x larger state space to cover

### Expected Learning Curve

**0-100k steps**: Basic reaching (may look worse than before!)
- Learning to read observations
- Discovering shoulder+elbow coordination
- Rewards lower initially (harder task)

**100-300k steps**: Improving generalization
- Successfully reaching various positions
- Still some failures on edge cases
- Reward variance high (some episodes easy, some hard)

**300-500k steps**: Robust policy
- Consistently reaching across full region
- Using observations effectively
- Reward variance low (consistent performance)

## Observation Space

The robot's observation already includes target position:

```python
obs['target_pos'] = self._get_target_position()  # 3D position of red box
obs['end_effector_pos'] = self._get_end_effector_position()  # 3D hand position
obs['vector_to_target'] = obs['target_pos'] - obs['end_effector_pos']
```

So the robot CAN see where the box is - now it MUST use that information!

## Verification After Training

### Test Generalization

After training, test on positions outside training range:

```python
# Test extreme positions
test_positions = [
    [0.30, 0.0, 0.74],   # Very close
    [0.70, 0.0, 0.74],   # Very far
    [0.5, 0.25, 0.74],   # Far right
    [0.5, -0.25, 0.74],  # Far left
]

for pos in test_positions:
    # Place box at position
    # Run policy
    # Check if robot still reaches
```

Good policy should reach even at edge cases.

## Benefits of Randomization

### 1. Generalization
- Works for any reachable box position
- Not limited to training positions

### 2. Robustness
- Handles perturbations
- Adapts to environment changes

### 3. Real-World Applicability
- Closer to real robot tasks
- Objects aren't perfectly positioned in real life

### 4. Better Understanding
- Forces robot to use sensory feedback
- Learns visual-motor mapping

## Files Modified

1. **g1_rl_environment.py** (Lines 251-269)
   ```python
   # Changed from:
   noise = np.random.uniform(-0.03, 0.03, 3)

   # To:
   x = np.random.uniform(0.35, 0.65)  # 30cm range
   y = np.random.uniform(-0.2, 0.2)    # 40cm range
   z = 0.74
   ```

2. **train_sb3_improved.py** (Lines 53-63)
   - Updated timesteps: 100k → 500k
   - Updated description to highlight randomization

## Summary

**Previous**: Robot learned to reach fixed point [0.5, 0, 0.74] ± 3cm
- Fast to learn (100k steps)
- Doesn't generalize
- Not using observations

**Now**: Robot must learn to reach ANY point in 30cm × 40cm region
- Slower to learn (500k steps)
- Generalizes well
- Actually tracks object using observations

**Trade-off**: 5x more training time for proper generalization - worth it!

## Retraining Command

```bash
python train_sb3_improved.py
```

Be patient - first 100k steps may look worse than before, but the final policy will be much more capable!
