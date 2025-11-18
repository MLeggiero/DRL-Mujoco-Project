#!/bin/bash
# Train G1 Reaching with Stable-Baselines3 PPO
# Professional, production-ready RL training

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to parent directory (fixed_torso_environment)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================="
echo "Stable-Baselines3 PPO Training"
echo "========================================="
echo ""

# Default: 100k timesteps (good for quick testing)
# For production training, use train_sb3_improved.py instead

python train_sb3.py \
  --scene ../unitree_g1/g1_table_box_scene.xml \
  --timesteps 100000 \
  --lr 3e-4 \
  --n_steps 2048 \
  --batch_size 64 \
  --n_epochs 10 \
  --seed 0

# Training features:
# - Automatic observation and reward normalization
# - Checkpoint saving every 10k steps
# - Evaluation every 5k steps
# - Best model tracking
# - TensorBoard logging
# - Progress bar
#
# After training completes, you can:
# 1. View training progress: tensorboard --logdir ./logs
# 2. Test best model: python train_sb3.py --test ./models/g1_ppo_*/best_model/best_model
