# Reward Structure V2 - Quick Reference

## 🎯 Goal: Break the 0.1m plateau and reach 0.05m success!

---

## Reward Zones (What the Robot "Feels" at Each Distance)

### 📍 At 0.20m - "Too Far Zone"
```
Exponential reward:     +0.05
Success bonus:          0
Comfort PENALTY:        -100  ⚠️ Very uncomfortable!
Proximity bonuses:      0
─────────────────────────────
Base reward:            -100  (strongly negative)
Message: "GET CLOSER NOW!"
```

### 📍 At 0.12m - "Old Plateau Zone"
```
Exponential reward:     +0.16
Success bonus:          0
Comfort PENALTY:        -80   ⚠️ Uncomfortable!
Proximity bonuses:      +50   (crossed 0.15m)
─────────────────────────────
Base reward:            -30   (negative - must move!)
Message: "Still too far, keep pushing!"
```

### 📍 At 0.10m - "Barrier Zone"
```
Exponential reward:     +0.22
Success bonus:          0
Comfort PENALTY:        -80   ⚠️ Uncomfortable!
Proximity bonuses:      +150  (0.15m + 0.10m milestones)
─────────────────────────────
Base reward:            +72   (positive, but penalty hurts)
Message: "Breaking through! Push to 0.08m!"
```

### 📍 At 0.08m - "Close Zone"
```
Exponential reward:     +0.30
Success bonus:          0
Comfort PENALTY:        -40   ⚠️ Mild pressure
Proximity bonuses:      +350  (0.15m + 0.10m + 0.08m)
─────────────────────────────
Base reward:            +310  (much better!)
Message: "Good progress! See +500 more at 0.06m!"
```

### 📍 At 0.06m - "Almost There Zone"
```
Exponential reward:     +0.41
Success bonus:          0
Comfort PENALTY:        -40   ⚠️ Mild pressure
Proximity bonuses:      +850  (all 4 milestones!)
─────────────────────────────
Base reward:            +810  (excellent!)
Message: "So close! +5000 waiting at goal!"
```

### 📍 At 0.05m - "SUCCESS ZONE" 🎯
```
Exponential reward:     +0.47
Success bonus:          +5000 ✨✨✨
Comfort PENALTY:        0     ✓ No penalty!
Proximity bonuses:      +850
─────────────────────────────
Base reward:            +5850 🎉🎉🎉
Message: "🎯 SUCCESS! GOAL REACHED!"
```

---

## Movement Rewards (Per Step)

### Approaching Target:
```python
Moving 1cm closer:  +2.0 reward  (200 * 0.01)
Moving 2cm closer:  +4.0 reward  (200 * 0.02)
Moving 5cm closer: +10.0 reward  (200 * 0.05)
```

### Moving Away from Target:
```python
Moving away:  -20.0 penalty  (discourages backtracking)
```

### Velocity Toward Target:
```python
Moving at 0.1 m/s toward:  +1.0 reward   (10 * 0.1)
Moving at 0.2 m/s toward:  +2.0 reward   (10 * 0.2)
Moving at 0.5 m/s toward:  +5.0 reward   (10 * 0.5)
```

---

## Complete Reward Formula

```python
total_reward = (
    exp(-15 * distance)                    # Exponential gradient
    + success_bonus                         # +5000 if < 0.05m
    + comfort_penalty                       # Tiered: -100, -80, -40
    + velocity_reward                       # +10 * v_toward
    + proximity_bonus                       # Cumulative milestones
    + approach_bonus                        # +200 * Δdistance
    + joint_velocity_penalty                # -0.005 * Σ(qvel²)
    + action_penalty                        # -0.005 * Σ(action²)
    + smoothness_penalty                    # -weight * Σ(Δaction²)
)
```

---

## Proximity Bonuses (Cumulative)

| Milestone | Individual Bonus | Cumulative Total |
|-----------|------------------|------------------|
| < 0.15m | +50 | +50 |
| < 0.10m | +100 | +150 |
| < 0.08m | +200 | +350 |
| < 0.06m | +500 | +850 |
| < 0.05m | +5000 (success) | +5850 |

---

## Tiered Comfort Penalties

| Distance Range | Penalty | Purpose |
|----------------|---------|---------|
| > 0.15m | **-100** | Break 0.3m plateau |
| 0.08m - 0.15m | **-80** | Break 0.1m plateau |
| 0.06m - 0.08m | **-40** | Push to finish |
| < 0.06m | **0** | No penalty near goal |

---

## Key Differences from V1

### What Changed:

1. **Exponential steepness**: -10 → **-15** (stronger near-goal gradient)
2. **Success bonus**: 1000 → **5000** (5x more attractive!)
3. **Comfort penalties**: Single @0.2m → **3 tiers** (break all plateaus)
4. **NEW: Proximity bonuses** (milestone rewards for progress)
5. **Approach multiplier**: 100 → **200** (doubled immediate feedback)

### Why It Breaks 0.1m Plateau:

**Before (V1):**
- At 0.1m: Small positive reward, no penalty
- Robot comfortable staying put

**After (V2):**
- At 0.1m: -80 penalty active!
- Can see +850 waiting at 0.06m
- Can see +5000 waiting at goal
- No comfortable stopping point

---

## Training Progress Indicators

### Breaking Through 0.1m:
- ✅ Distance consistently < 0.10m
- ✅ Comfort penalty showing -80 in logs
- ✅ Proximity bonuses appearing (+150)

### Reaching 0.08m:
- ✅ Distance consistently < 0.08m
- ✅ Comfort penalty reduced to -40
- ✅ Proximity bonuses showing +350

### Approaching Goal:
- ✅ Distance < 0.06m appearing
- ✅ Proximity bonuses showing +850
- ✅ Episodes ending closer to target

### SUCCESS! 🎯
- ✅ "🎯 SUCCESS!" messages in logs
- ✅ Episodes reaching < 0.05m
- ✅ Huge reward spikes to +5000+
- ✅ Success rate increasing

---

## Debugging Tips

### If still stuck at 0.1m after 200k steps:

1. **Check comfort penalty is active:**
   - Look for penalty values around -80 to -100 in rewards
   - If not seeing penalties, check distance measurements

2. **Verify proximity bonuses:**
   - Should see +150 when crossing 0.1m threshold
   - These are one-time per episode

3. **Watch approach bonus:**
   - Should be positive when distance decreasing
   - Should be ~+2.0 per cm of progress

4. **Consider increasing action scale:**
   - Current: 0.5
   - Try: 0.6 or 0.7 for even finer control

### If reaching 0.06m but not 0.05m:

This is good progress! The robot is very close. Try:
- Increase action scale to 0.7
- Increase success bonus to 10000
- Train longer (patience - it's learning fine control)

---

## Expected Training Curve

Based on your current progress (1M steps at 0.1m):

```
Steps      Distance    Status
─────────────────────────────────────
0 - 300k   0.3m       Initial plateau
300k - 1M  0.1m       ✅ First breakthrough
1M - 1.2M  0.08m      ⏳ Expect this next
1.2M - 1.5M 0.06m     ⏳ Then this
1.5M+      < 0.05m    ⏳ Success! 🎯
```

You're already at 1M, so the next 500k steps should show dramatic improvement!

---

## Success Criteria

Your training is succeeding when you see:
1. ✅ Moving average distance consistently dropping
2. ✅ Regular -80 to -100 penalties in reward logs
3. ✅ Proximity bonuses appearing (+150, +350, +850)
4. ✅ Success messages: "🎯 SUCCESS!"
5. ✅ Final distance plot showing values at/below red line (0.05m)

Good luck! The rewards are now properly structured to guide your robot all the way to the goal! 🚀
