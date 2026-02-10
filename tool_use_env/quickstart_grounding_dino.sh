#!/bin/bash
# Quick start script for Grounding DINO + RL grasping

echo "=================================================="
echo "Grounding DINO + RL Grasping - Quick Start"
echo "=================================================="

# 1. Test environment
echo ""
echo "Step 1: Testing Grounding DINO environment..."
python grounding_dino_grasp_env.py

if [ $? -ne 0 ]; then
    echo "❌ Environment test failed!"
    exit 1
fi

echo ""
echo "✅ Environment test passed!"

# 2. Start training (quick test)
echo ""
echo "Step 2: Starting quick training run..."
echo "(Training for 10K steps with 2 environments)"
echo ""

python train_grounding_dino_grasp.py \
    --prompt "hammer" \
    --num-envs 2 \
    --timesteps 10000 \
    --output-dir models/grounding_dino_test

echo ""
echo "=================================================="
echo "Quick start complete!"
echo "=================================================="
echo ""
echo "To train for real:"
echo "  python train_grounding_dino_grasp.py --timesteps 1000000 --num-envs 4"
echo ""
echo "To test different objects:"
echo "  python train_grounding_dino_grasp.py --prompt 'screwdriver'"
echo "  python train_grounding_dino_grasp.py --prompt 'red tool'"
echo ""
echo "To use vision-based rewards (harder):"
echo "  python train_grounding_dino_grasp.py --use-vision"
echo ""
