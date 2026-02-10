#!/bin/bash
# Install Grounding DINO for object detection

echo "Installing Grounding DINO dependencies..."

# Install groundingdino from source
pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git#egg=groundingdino

# Or use the pip package (simpler but may be older)
# pip install groundingdino-py

# Install required dependencies
pip install transformers
pip install huggingface_hub
pip install supervision

echo "✓ Installation complete!"
echo "Run: python grounding_dino_detector.py to test"
