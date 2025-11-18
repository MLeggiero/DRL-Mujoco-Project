#!/bin/bash
# Quick visualization script for trained policies

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to parent directory (fixed_torso_environment)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================="
echo "G1 Policy Visualization"
echo "========================================="
echo ""

# Check for command line argument
if [ "$1" == "--interrupted" ] || [ "$1" == "-i" ]; then
    # Find most recent interrupted model
    LATEST_MODEL=$(find ./models -name "interrupted_model.zip" -type f | sort -r | head -1)
    MODEL_TYPE="interrupted"
else
    # Find most recent best model
    LATEST_MODEL=$(find ./models -name "best_model.zip" -type f | sort -r | head -1)
    MODEL_TYPE="best"
fi

if [ -z "$LATEST_MODEL" ]; then
    echo "No $MODEL_TYPE models found!"
    echo ""
    echo "Options:"
    echo "  1. Train a model first: python train_sb3_improved.py"
    echo "  2. Visualize interrupted model: ./scripts/visualize.sh --interrupted"
    echo "  3. Visualize random policy: python visualize_policy.py --random"
    exit 1
fi

# Remove .zip extension for model path
LATEST_MODEL_PATH="${LATEST_MODEL%.zip}"

echo "Found latest $MODEL_TYPE model:"
echo "  $LATEST_MODEL"
echo ""
echo "Launching MuJoCo viewer..."
echo ""

# Run visualization
python visualize_policy.py \
    --model "$LATEST_MODEL_PATH" \
    --episodes 5 \
    --slow

echo ""
echo "========================================="
echo "Visualization complete!"
echo "========================================="
