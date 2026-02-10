# Grounding DINO Object Detection & Geometry Guide

## Overview

Grounding DINO is a zero-shot object detector that uses natural language prompts. Unlike traditional object detectors (like YOLO) that are limited to pre-trained classes, Grounding DINO can detect ANY object you describe with text.

## Quick Start

### 1. Install Dependencies

```bash
bash install_grounding_dino.sh
```

Or manually:
```bash
pip install groundingdino-py
pip install transformers huggingface_hub supervision
```

### 2. Basic Usage

```python
from grounding_dino_detector import GroundingDINODetector
import cv2

# Initialize detector (downloads model on first run)
detector = GroundingDINODetector()

# Load image
rgb = cv2.imread("your_image.png")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

# Detect objects using natural language
detections = detector.detect(
    rgb,
    text_prompt="hammer",  # What to find
    box_threshold=0.25     # Confidence threshold
)

# Print results
for det in detections:
    print(f"Found {det['label']} at {det['bbox']} with {det['confidence']:.2%} confidence")
```

### 3. Run Example

```bash
python simple_grounding_dino_example.py
```

## Features

### 1. Object Detection

Detect objects using natural language prompts:

```python
# Single object
detections = detector.detect(rgb, text_prompt="hammer")

# Be more specific
detections = detector.detect(rgb, text_prompt="red hammer with wooden handle")

# Detect parts
detections = detector.detect(rgb, text_prompt="hammer head")
```

### 2. Get 2D Geometry

Each detection includes:
- `bbox`: Bounding box [x1, y1, x2, y2] in pixels
- `bbox_normalized`: Normalized coordinates [0-1]
- `center`: Center point [cx, cy]
- `confidence`: Detection confidence [0-1]

```python
det = detections[0]
x1, y1, x2, y2 = det['bbox']

# Calculate size
width = x2 - x1
height = y2 - y1
print(f"Object size: {width} x {height} pixels")

# Get center
cx, cy = det['center']
```

### 3. Get 3D Position (with Depth)

Convert 2D detection to 3D using depth image:

```python
import numpy as np

# Load depth image (in meters)
depth = load_depth_image()  # Shape: (H, W)

# Camera intrinsics (from camera calibration)
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

# Get 3D position
pos_3d = detector.get_3d_position(detections[0], depth, K)
print(f"3D position: {pos_3d}")  # [x, y, z] in meters
```

### 4. Visualize Results

```python
# Draw bounding boxes on image
vis_image = detector.visualize_detections(
    rgb,
    detections,
    save_path="output.png"
)
```

## Parameters

### `detect()` Parameters

- **text_prompt**: What to detect (e.g., "hammer", "red tool", "metal object")
- **box_threshold**: Confidence threshold (default: 0.35)
  - Lower (0.2-0.3): More detections, may include false positives
  - Higher (0.4-0.6): Fewer detections, more confident
- **text_threshold**: Text matching threshold (default: 0.25)

### Camera Intrinsics

For 3D geometry, you need camera intrinsics `K`:

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Where:
- `fx, fy`: Focal lengths (pixels)
- `cx, cy`: Principal point (usually image center)

**Common values:**
- 640x480 camera: fx=fy=525, cx=320, cy=240
- 1920x1080 camera: fx=fy=1050, cx=960, cy=540

Or measure from your camera using calibration.

## Advanced Usage

### Detect Multiple Object Types

```python
# Combine multiple prompts
detections = detector.detect(rgb, text_prompt="hammer . screwdriver . wrench")

# Or run separately
for obj in ["hammer", "nail", "screw"]:
    dets = detector.detect(rgb, text_prompt=obj)
    print(f"{obj}: {len(dets)} found")
```

### Get Object Orientation (Bounding Box Method)

```python
# Estimate orientation from bounding box aspect ratio
x1, y1, x2, y2 = det['bbox']
width = x2 - x1
height = y2 - y1

# For elongated objects (hammer, screwdriver)
if width > height:
    orientation = "horizontal"
else:
    orientation = "vertical"

aspect_ratio = max(width, height) / min(width, height)
print(f"Aspect ratio: {aspect_ratio:.2f}")
```

### Filter by Confidence

```python
# Get high-confidence detections only
confident_dets = [d for d in detections if d['confidence'] > 0.5]

# Get best detection
best = max(detections, key=lambda x: x['confidence'])
```

## Tips for Better Detection

### 1. Text Prompts

**Good prompts:**
- "hammer" - Simple and clear
- "claw hammer" - Specific type
- "red hammer" - Include color
- "metal hammer head" - Describe parts

**Avoid:**
- Too vague: "tool"
- Too complex: "the red hammer with a wooden handle that is lying on the table"

### 2. Thresholds

Start with:
- `box_threshold=0.25` for initial exploration
- `box_threshold=0.35` for production use
- `box_threshold=0.45+` for very confident detections

### 3. Image Quality

Better results with:
- Good lighting
- Clear view of object
- Minimal occlusion
- Sufficient resolution (640x480 or higher)

## Comparison with YOLO

| Feature | Grounding DINO | YOLO |
|---------|---------------|------|
| Object types | ANY (via text) | Fixed 80 classes |
| Training | Pre-trained | Pre-trained |
| Speed | Slower (~1-2 FPS) | Faster (~30+ FPS) |
| Accuracy | High for novel objects | High for trained classes |
| Use case | Research, novel objects | Real-time, known objects |

**When to use Grounding DINO:**
- Detecting objects not in YOLO's 80 classes
- Don't want to train a custom model
- Need flexibility with text prompts
- Research/prototyping

**When to use YOLO:**
- Real-time detection needed
- Detecting common objects (person, car, etc.)
- Production deployment

## Example: Full Pipeline

```python
import cv2
import numpy as np
from grounding_dino_detector import GroundingDINODetector

# Initialize
detector = GroundingDINODetector()

# Load data
rgb = cv2.imread("scene.png")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
depth = np.load("depth.npy")  # In meters

# Detect
detections = detector.detect(rgb, text_prompt="hammer", box_threshold=0.3)

if len(detections) > 0:
    # Best detection
    best = detections[0]

    # 2D info
    print(f"Confidence: {best['confidence']:.2%}")
    print(f"Bounding box: {best['bbox']}")

    # 3D position (requires camera intrinsics)
    K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]])
    pos_3d = detector.get_3d_position(best, depth, K)
    print(f"3D position: {pos_3d}")

    # Visualize
    detector.visualize_detections(rgb, detections, "output.png")
```

## Troubleshooting

### "No detections found"
- Lower `box_threshold` (try 0.2)
- Try different text prompts
- Check image quality

### "CUDA out of memory"
- Use CPU: Set `device="cpu"` in detector initialization
- Reduce image size before detection

### "Model download failed"
- Check internet connection
- May need to manually download from HuggingFace
- Firewall/proxy issues

## Resources

- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)
- [GitHub Repository](https://github.com/IDEA-Research/GroundingDINO)
- [HuggingFace Model](https://huggingface.co/ShilongLiu/GroundingDINO)
