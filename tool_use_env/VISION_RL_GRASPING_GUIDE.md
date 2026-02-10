
# Vision + RL Grasping - Complete Guide

## Overview

This guide shows you how to combine **multi-object vision detection** with **reinforcement learning** to train robots to grasp objects.

**The Pipeline:**
```
Camera → Vision Detection → Target Position → RL Policy → Robot Action → Grasp Success
```

---

## Quick Start

### 1. Test the Environment

```bash
python vision_guided_grasp_env.py
```

### 2. Start Training (Recommended: Hybrid Strategy)

```bash
python train_vision_grasp.py --strategy hybrid --timesteps 1000000
```

### 3. Monitor Progress

```bash
tensorboard --logdir models/vision_grasp/hybrid/tensorboard
```

---

## Training Strategies

We provide **4 different strategies** from easiest to hardest:

### Strategy 1: Physics Baseline (Easiest, Fastest)

**What**: Uses ground truth object positions from physics engine
**Why**: Validate RL setup, fast convergence
**When**: First step, debugging

```bash
python train_vision_grasp.py --strategy baseline --timesteps 500000
```

**Expected Results:**
- Training time: ~30 minutes (4 envs)
- Success rate: 80-90%
- Use case: Baseline comparison

###
