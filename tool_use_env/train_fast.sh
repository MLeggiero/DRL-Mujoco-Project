#!/bin/bash
#
# Fast Training Script for Vision-Guided Grasping
#
# This script uses optimized settings to speed up training:
# - Vision updates every 10 episodes instead of every episode
# - 4 parallel environments
# - This should train 10-20x faster than default settings
#

echo "=================================================="
echo "  Fast Vision-Guided Grasping Training"
echo "=================================================="
echo ""
echo "Optimizations:"
echo "  - Vision detection: Every 10 episodes (not every episode)"
echo "  - Parallel envs: 4"
echo "  - Expected speed: ~300-500 it/s (vs ~30 it/s default)"
echo "  - Expected time: ~1 hour (vs ~10 hours default)"
echo ""
echo "=================================================="
echo ""

python train_vision_grasp.py \
    --strategy hybrid \
    --timesteps 1000000 \
    --num-envs 4 \
    --target hammer \
    --vision-freq 10

echo ""
echo "=================================================="
echo "  Training Complete!"
echo "=================================================="
echo ""
echo "Model saved to: models/vision_grasp/hybrid/"
echo ""
echo "To monitor with TensorBoard:"
echo "  tensorboard --logdir models/vision_grasp/hybrid/tensorboard"
echo ""
