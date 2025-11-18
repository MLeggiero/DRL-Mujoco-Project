# Training Improvements Guide

Based on the first training run (100k steps, best reward: -719), here are concrete recommendations for improvement.

## Analysis of First Training Run

### Results Summary
- **Best reward**: -719 (at 30k steps)
- **Final reward**: -863 (at 100k steps)
- **Episode length**: 1000 steps (always hitting max)
- **Status**: Learning plateaued after 30k steps

### Key Observations
1. ✅ Training was stable (no crashes)
2. ✅ Gradual improvement in first 30k steps
3. ⚠️ Plateaued or regressed after 30k steps
4. ❌ Never completed an episode early (never reached target)
5. ❌ Episodes hitting 1000-step limit suggests task is too long

## Priority Improvements

### 🔥 Priority 1: Reduce Episode Length

**Problem**: 1000 steps is excessive for a reaching task. This causes:
- Massive time penalty accumulation (-0.01 × 1000 = -10)
- Slow learning (robot doesn't see episode endings)
- Inefficient exploration

**Solution**: Reduce to 300-500 steps

```python
# In g1_rl_environment.py, line ~27
self.max_episode_steps = 300  # Was 1000
```

Or register environment with custom max steps:
```python
# In g1_gym_wrapper.py
gym.register(
    id='G1Reaching-v0',
    entry_point='g1_gym_wrapper:G1ReachingGymEnv',
    max_episode_steps=300,  # Was 1000
)
```

**Expected impact**: ⭐⭐⭐ (Critical - will dramatically improve learning)

---

### 🔥 Priority 2: Reduce Time Penalty

**Problem**: Current time penalty (-0.01 per step) dominates reward signal

**Solution**: Reduce time penalty

```python
# In g1_rl_environment.py, _calculate_reward method
time_penalty = -0.001  # Was -0.01 (10x reduction)
```

**Expected impact**: ⭐⭐⭐ (Critical - makes distance reward more prominent)

---

### 🔥 Priority 3: Increase Training Time

**Problem**: 100k steps may not be enough for this complex task

**Solution**: Train for 300k-500k steps
```bash
python train_sb3.py --timesteps 300000
```

Or use the improved config:
```bash
python train_sb3_improved.py
```

**Expected impact**: ⭐⭐ (Important - gives agent more time to learn)

---

### 🔥 Priority 4: Improve Reward Shaping

**Problem**: Current reward might not provide enough signal for getting close

**Current reward structure**:
```python
distance_reward = -distance          # Main signal
progress_reward = progress * 10.0    # Bonus for improvement
proximity_bonus = 0.5 (if < 0.3m)    # Small bonuses
                 + 1.0 (if < 0.15m)
                 + 5.0 (if < 0.05m)
time_penalty = -0.01                  # Per-step cost
```

**Improved reward structure**:
```python
# Stronger shaped reward
distance_reward = -distance

# Exponential proximity reward (gets much larger when close)
if distance < 0.5:
    proximity_bonus = 10.0 * (0.5 - distance)  # 0 to 5.0
if distance < 0.2:
    proximity_bonus += 20.0 * (0.2 - distance)  # Additional 0 to 4.0
if distance < 0.05:  # Success
    proximity_bonus += 50.0  # Large success bonus

# Much smaller time penalty
time_penalty = -0.001  # 10x smaller

# Optional: velocity reward (encourage faster reaching)
hand_velocity = np.linalg.norm(self.data.qvel[hand_indices])
velocity_reward = 0.01 * hand_velocity if distance > 0.1 else 0
```

**Expected impact**: ⭐⭐⭐ (Critical - provides better learning signal)

---

### Priority 5: Hyperparameter Tuning

**Problem**: Default hyperparameters might not be optimal

**Recommended changes**:

```python
# More aggressive learning
learning_rate = 5e-4      # Was 3e-4 (higher = faster learning)

# More frequent updates
n_steps = 1024            # Was 2048 (update more often)

# Larger batches for stability
batch_size = 128          # Was 64 (more stable gradients)

# More gradient steps per update
n_epochs = 20             # Was 10 (learn more from each batch)

# Encourage more exploration
ent_coef = 0.02           # Was 0.01 (more randomness)
```

Use the improved config:
```bash
python train_sb3_improved.py
```

**Expected impact**: ⭐⭐ (Moderate - can speed up learning)

---

### Priority 6: Curriculum Learning

**Problem**: Task might be too hard initially (target randomization)

**Solution**: Start easy, gradually increase difficulty

```python
# In g1_rl_environment.py, _randomize_object_positions
def _randomize_object_positions(self):
    # Gradually increase randomization over training
    max_noise = min(0.03, self.current_step / 100000 * 0.03)  # 0 to 3cm

    for obj_name in self.target_objects:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id >= 0:
            for i in range(self.model.njnt):
                if self.model.jnt_bodyid[i] == body_id:
                    qpos_addr = self.model.jnt_qposadr[i]
                    noise = np.random.uniform(-max_noise, max_noise, 3)
                    self.data.qpos[qpos_addr:qpos_addr+3] += noise
                    break
```

**Expected impact**: ⭐ (Minor - helps initial learning)

---

## Quick Wins - Do These First

### 1. Test Current Best Model
```bash
python train_sb3.py --test ./models/g1_ppo_20251117_133846/best_model/best_model
```
See actual performance before making changes.

### 2. Reduce Episode Length
Edit `g1_rl_environment.py` line 27:
```python
self.max_episode_steps = 300  # Was 1000
```

### 3. Reduce Time Penalty
Edit `g1_rl_environment.py` line 384:
```python
time_penalty = -0.001  # Was -0.01
```

### 4. Retrain with Improved Config
```bash
python train_sb3_improved.py
```

This alone should give you **much better results**.

---

## Advanced Improvements

### Option 1: Add Shaped Distance Reward

Instead of linear `-distance`, use exponential:
```python
# Exponential reward - gets much better when close
distance_reward = -np.exp(distance) + 1  # Range: ~0 (far) to 1 (close)
```

### Option 2: Add Hindsight Experience Replay (HER)

SB3 supports HER for sparse reward tasks:
```python
from stable_baselines3 import HerReplayBuffer, SAC

model = SAC(
    "MlpPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
    ),
)
```

### Option 3: Try Different Algorithms

PPO might not be best for continuous control:
```python
# Try SAC (often better for robot manipulation)
from stable_baselines3 import SAC

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300_000)
```

### Option 4: Normalize Actions

Current actions are clipped to [-1, 1] then scaled by 0.5. This wastes action space:
```python
# In g1_rl_environment.py, _apply_action
# Remove the 0.5 scaling, let agent use full range
self.data.ctrl[actuator_id] = action[i]  # Was: action[i] * 0.5
```

Then tune the actuator gains in the XML file if needed.

---

## Recommended Implementation Plan

### Phase 1: Quick Fixes (10 minutes)
1. Reduce episode length to 300 steps
2. Reduce time penalty to -0.001
3. Retrain with improved config: `python train_sb3_improved.py`

**Expected result**: Best reward improves from -719 to -200 to -100

### Phase 2: Reward Tuning (30 minutes)
1. Implement exponential proximity reward
2. Add velocity reward
3. Test and iterate

**Expected result**: Agent starts reaching target occasionally

### Phase 3: Advanced (1-2 hours)
1. Implement curriculum learning
2. Try SAC algorithm
3. Tune network architecture

**Expected result**: Consistent success (>50% success rate)

---

## Monitoring Progress

### Key Metrics to Watch

**During Training**:
- `eval/mean_reward`: Should increase over time
- `eval/mean_ep_length`: Should decrease (episodes ending earlier = reaching target)
- `train/explained_variance`: Should be > 0.5 (model understanding value function)

**In TensorBoard**:
```bash
tensorboard --logdir ./logs
```

Watch for:
- Reward curve trending upward
- Episode length decreasing
- Value loss stabilizing

### Success Criteria

**Baseline (current)**:
- Reward: -719
- Distance: ~0.7-0.9m
- Success: 0%

**Good progress**:
- Reward: -100 to -50
- Distance: ~0.2-0.3m
- Success: ~10%

**Excellent**:
- Reward: 0 to +5
- Distance: <0.1m
- Success: >50%

---

## Expected Timeline

### Conservative Estimate
- Quick fixes: +1 training run (~30 min)
- Reward tuning: +2-3 training runs (~2 hours)
- Advanced tuning: +5-10 runs (~1 day)

**Total: 1-2 days to achieve 50%+ success rate**

### With Good Luck
- Quick fixes alone might get 20-30% success
- Total: 2-3 hours to good performance

---

## Files to Modify

### Critical (do these first):
1. `g1_rl_environment.py` - Episode length, time penalty
2. Use `train_sb3_improved.py` - Better hyperparameters

### Optional (for further improvement):
3. `g1_rl_environment.py` - Reward function (proximity bonus)
4. `train_sb3.py` - Try different algorithms (SAC)

---

## Next Steps

1. **Test current model**: See baseline performance
   ```bash
   python train_sb3.py --test ./models/g1_ppo_20251117_133846/best_model/best_model
   ```

2. **Apply quick fixes**: Episode length + time penalty

3. **Retrain**: Use improved config
   ```bash
   python train_sb3_improved.py
   ```

4. **Monitor**: Check TensorBoard

5. **Iterate**: Based on results, apply advanced improvements

The quick fixes alone should give you a **2-5x improvement** in performance. Combined with more training time, you should see the agent successfully reaching the target within a few training runs.
