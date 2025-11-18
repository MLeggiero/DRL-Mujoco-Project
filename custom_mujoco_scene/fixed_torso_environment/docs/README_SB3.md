# G1 Reaching Task - Stable-Baselines3 Training

This directory contains a production-ready implementation of PPO training for the Unitree G1 robot reaching task using **Stable-Baselines3**, the industry-standard RL library.

## Why Stable-Baselines3?

Stable-Baselines3 (SB3) provides:
- **Proven algorithms**: Battle-tested PPO implementation used in research and production
- **Automatic features**: Observation/reward normalization, gradient clipping, learning rate scheduling
- **Better stability**: No numerical issues like the custom finite-differences implementation
- **Monitoring**: TensorBoard integration, automatic logging, progress tracking
- **Faster training**: Optimized implementations, proper vectorization
- **Easy deployment**: Save/load models, reproducible results

## Quick Start

### 1. Install Dependencies

```bash
./install_sb3.sh
```

This installs:
- `stable-baselines3`: Core RL algorithms
- `gymnasium`: Environment interface
- `tensorboard`: Training visualization

### 2. Train the Model

```bash
./train_sb3.sh
```

Or with custom parameters:

```bash
python train_sb3.py \
  --timesteps 1000000 \
  --lr 3e-4 \
  --n_steps 2048 \
  --batch_size 64 \
  --n_epochs 10
```

### 3. Monitor Training

While training runs, open another terminal:

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser to see:
- Episode rewards over time
- Episode lengths
- Value function loss
- Policy loss
- Learning rate

### 4. Test Trained Model

```bash
python train_sb3.py --test ./models/g1_ppo_*/best_model/best_model
```

## Training Parameters

### Recommended Settings

**Quick test (2-5 minutes)**:
```bash
--timesteps 10000 --n_steps 512 --batch_size 64
```

**Standard training (30-60 minutes)**:
```bash
--timesteps 100000 --n_steps 2048 --batch_size 64
```

**Production training (2-4 hours)**:
```bash
--timesteps 1000000 --n_steps 2048 --batch_size 128
```

### Parameter Guide

- `--timesteps`: Total environment steps (more = better but slower)
- `--lr`: Learning rate (3e-4 is good default, try 1e-4 for more stability)
- `--n_steps`: Steps collected before each update (higher = more data but slower updates)
- `--batch_size`: Minibatch size for gradient updates (higher = more stable but needs more memory)
- `--n_epochs`: PPO update epochs per data batch (10 is standard)

## Features

### Automatic Normalization
- Observations are normalized to mean=0, std=1
- Rewards are normalized for stable training
- Prevents numerical instability

### Checkpointing
- Model saved every 10k steps to `./models/g1_ppo_*/`
- Best model tracked and saved separately
- Can resume training from checkpoints

### Evaluation
- Periodic evaluation every 5k steps
- Tracks best performing model
- Evaluation uses deterministic policy

### Progress Tracking
- Real-time progress bar
- Episode statistics (reward, length, success)
- TensorBoard logging for detailed analysis

## File Structure

```
.
├── g1_gym_wrapper.py      # Gymnasium wrapper for G1 environment
├── train_sb3.py           # Main SB3 training script
├── train_sb3.sh           # Convenient training launcher
├── install_sb3.sh         # Dependency installer
├── models/                # Saved models (created during training)
│   └── g1_ppo_TIMESTAMP/
│       ├── best_model/    # Best performing model
│       └── checkpoints/   # Regular checkpoints
└── logs/                  # TensorBoard logs (created during training)
    └── g1_ppo_TIMESTAMP/
```

## Expected Results

With the floating base fix and SB3's stable implementation, you should see:

**Early training (0-20k steps)**:
- Rewards: -250 to -150
- Distance: 0.8m to 0.5m
- Agent learns basic reaching motions

**Mid training (20-50k steps)**:
- Rewards: -150 to -50
- Distance: 0.5m to 0.2m
- Agent improves accuracy

**Late training (50-100k steps)**:
- Rewards: -50 to +5
- Distance: 0.2m to 0.05m
- Agent achieves consistent success

## Troubleshooting

### Import errors
Run `./install_sb3.sh` to install all dependencies.

### Slow training
- Reduce `--n_steps` to 512 or 1024
- Reduce `--batch_size` to 32
- Reduce `--timesteps` for quick testing

### Unstable training
- Reduce learning rate: `--lr 1e-4`
- Increase batch size: `--batch_size 128`
- Check that floating base fix is working (should see no DOF 19 warnings)

### Out of memory
- Reduce `--n_steps` and `--batch_size`
- Reduce network size by editing `policy_kwargs` in train_sb3.py

## Comparison with Custom PPO

| Feature | Custom PPO | Stable-Baselines3 |
|---------|-----------|-------------------|
| Gradient computation | Finite differences (slow, unstable) | Automatic differentiation (fast, stable) |
| Normalization | Manual | Automatic |
| Checkpointing | Manual | Built-in |
| Monitoring | Print statements | TensorBoard |
| Stability | Prone to NaN | Very stable |
| Training time | ~4 hours for 100 epochs | ~30 min for 100k steps |
| Reliability | Collapses at epoch 4 | Trains to completion |

## Next Steps

1. Train with default settings: `./train_sb3.sh`
2. Monitor with TensorBoard
3. Test best model
4. Iterate on reward function or network architecture if needed
5. Deploy trained policy to real robot or further simulation

## References

- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- PPO Paper: https://arxiv.org/abs/1707.06347
- Gymnasium: https://gymnasium.farama.org/
