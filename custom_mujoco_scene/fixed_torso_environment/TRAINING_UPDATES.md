# Training Configuration Updates - Nov 17, 2025

## Critical Environment Fixes Applied

### 1. Objects Now Spawn Correctly
**Problem**: Objects were spawning at robot center (0.45m away), making task artificially easy but physically wrong.

**Fix**: Objects now spawn on table at correct XML positions:
- Red box: 0.70m forward, on table surface (0.74m height)
- Blue box: 0.40m forward, on table
- Green box: 0.35m forward, 0.25m to side, on table

**Impact**: Task is now 67% harder (0.75m vs 0.45m distance)

### 2. Legs Are Perfectly Locked
**Problem**: Legs were flailing despite zero actuator torques.

**Fix**: Position and velocity locking enforced before AND after each physics step.

**Result**: Zero leg movement - only arm and torso move.

### 3. Action Scaling Reduced
**Previous**: 0.5 (still too aggressive)
**Current**: 0.01 (ultra-gentle, stable)

**Impact**: Robot moves much more slowly and smoothly, but requires more training time.

## Updated Training Configuration

### Old Configuration (500k steps)
```python
{
    "total_timesteps": 500_000,
    "max_episode_steps": 300,
    "action_scaling": 0.5,
    # Trained on broken environment
}
```

### New Configuration (1M steps)
```python
{
    "total_timesteps": 1_000_000,  # 2x increase
    "max_episode_steps": 500,       # 67% increase
    "action_scaling": 0.01,         # 50x reduction
    "learning_rate": 5e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 20,
}
```

## Why More Timesteps Are Needed

### Task Difficulty Increased
1. **Distance**: 0.45m → 0.75m (67% harder)
2. **Precision required**: Objects now at realistic positions
3. **No shortcuts**: Can't exploit broken physics

### Movement Speed Reduced
1. **Action scaling**: 0.5 → 0.01 (50x slower)
2. **Episode length**: 300 → 500 steps (to accommodate slow movement)
3. **Exploration**: Takes longer to discover effective actions

### Estimated Training Time
- **100k steps**: Initial exploration, random reaching
- **300k steps**: Learning forward motion, distance decreasing
- **500k steps**: Refining reach accuracy
- **700k-1M steps**: Fine-tuning, achieving touches

## Expected Training Results

### Early Training (0-200k steps)
- **Reward**: -400 to -300
- **Distance**: 0.75m → 0.5-0.6m
- **Behavior**: Arm moves forward hesitantly

### Mid Training (200k-600k steps)
- **Reward**: -300 to -150
- **Distance**: 0.5m → 0.2-0.3m
- **Behavior**: Purposeful reaching, getting close

### Late Training (600k-1M steps)
- **Reward**: -150 to -50 (hopefully)
- **Distance**: 0.2m → 0.05-0.1m
- **Behavior**: Accurate reaching, some touches

### Success Metrics
- **Good progress**: Final distance < 0.15m
- **Excellent progress**: Final distance < 0.08m, occasional touches
- **Success**: Regular touches (< 0.05m), success rate > 10%

## Monitoring Training

### Watch These Metrics
```bash
tensorboard --logdir ./logs
```

Key metrics:
- `rollout/ep_rew_mean` - Should increase (less negative)
- `eval/mean_reward` - Should increase
- Distance to target - Should decrease

### Checkpoints to Check
Visualize these checkpoints to see progress:
```bash
# Early (200k)
python visualize_policy.py --model ./models/.../g1_ppo_checkpoint_200000_steps

# Mid (500k)
python visualize_policy.py --model ./models/.../g1_ppo_checkpoint_500000_steps

# Final (1M)
python visualize_policy.py --model ./models/.../best_model/best_model
```

## If Training Is Too Slow

### After 200k steps, if distance > 0.6m:
Action scaling might be too gentle. Try:
```python
# In g1_rl_environment.py line 293
self.data.ctrl[actuator_id] = action[i] * 0.02  # Increase to 0.02
```

Then retrain from scratch.

### After 500k steps, if distance > 0.3m:
- Increase timesteps to 2M
- Or increase learning rate to 1e-3
- Or increase action scaling to 0.03

## Starting Fresh Training

All old models have been deleted (trained on broken environment).

To start training:
```bash
python train_sb3_improved.py
```

This will train for 1M timesteps with the corrected environment.

Expected training time:
- **CPU**: 12-24 hours
- **GPU (if available)**: 3-6 hours

## Files Modified

1. `g1_rl_environment.py`
   - Line 28: `max_episode_steps = 500` (was 300)
   - Line 190-198: Only modify robot floating base, not objects
   - Line 274-294: Double-lock legs before AND after physics step
   - Line 293: `action_scaling = 0.01` (was 0.5)

2. `train_sb3_improved.py`
   - Line 14: `total_timesteps = 1_000_000` (was 500k)
   - Line 20: `n_steps = 2048` (was 1024)
   - Line 25: `gamma = 0.99` (was 0.98)

## Next Steps

1. **Start training**: `python train_sb3_improved.py`
2. **Monitor progress**: `tensorboard --logdir ./logs`
3. **Check at 200k**: Visualize checkpoint, should see forward motion
4. **Check at 500k**: Should be getting close to objects
5. **Final at 1M**: Should have learned to reach accurately

Good luck! The task is now physically correct and achievable.
