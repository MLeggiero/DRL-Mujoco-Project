#!/bin/bash
# Quick start script for RL-based grasping

set -e  # Exit on error

echo "=========================================="
echo "RL Grasping Quick Start"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -f "pointcloud_grasp_env.py" ]; then
    echo "ERROR: Please run from tool_use_env directory"
    exit 1
fi

# Function to run a step
run_step() {
    echo ""
    echo ">>> $1"
    echo ""
}

# Step 1: Test environment
run_step "Step 1/4: Testing point cloud environment..."
python pointcloud_grasp_env.py

if [ $? -ne 0 ]; then
    echo "ERROR: Environment test failed. Check dependencies."
    exit 1
fi

# Step 2: Generate initial point cloud for visualization
run_step "Step 2/4: Generating point cloud for visualization..."
python generate_pointcloud.py --camera track_front --output ./pointcloud_data

# Step 3: Detect grasps to establish baseline
run_step "Step 3/4: Detecting baseline grasps..."
python segment_hammer_spatial.py --pointcloud ./pointcloud_data/pointcloud.npz
python analyze_grasps.py --pointcloud ./pointcloud_data/hammer_segmented.npz --num-grasps 5

# Step 4: Ask user if they want to start training
run_step "Step 4/4: Ready to train!"
echo "Environment is ready! You can now:"
echo ""
echo "1. Start training (recommended):"
echo "   python train_pointcloud_grasp.py train --total-timesteps 1000000"
echo ""
echo "2. Or train with more parallel environments:"
echo "   python train_pointcloud_grasp.py train --num-envs 8 --total-timesteps 2000000"
echo ""
echo "3. Monitor training in another terminal:"
echo "   tensorboard --logdir ./training_output/tensorboard"
echo ""
echo "4. Or use existing state-based RL (faster, simpler):"
echo "   python train_rgbd_sb3.py"
echo ""
read -p "Start training now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting training with default settings..."
    echo "This will take several hours. Press Ctrl+C to stop."
    sleep 2
    python train_pointcloud_grasp.py train --total-timesteps 1000000
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Read RL_GRASPING_GUIDE.md for detailed instructions"
