# Using Learned Models for Hammer Recognition

**Yes!** Learned vision models are the modern, robust approach for object recognition in robotics.

---

## 🎯 Available Options

### **Option 1: Foundation Models (Zero-Shot)** ⭐ Easiest

Use pre-trained models that work without any training on your data.

#### **A. Grounding DINO** (Text → Object Detection)
```python
from groundingdino.util.inference import load_model, predict

# Load pre-trained model
model = load_model("path/to/GroundingDINO_SwinT_OGC.pth")

# Detect with natural language
boxes, logits, phrases = predict(
    model=model,
    image=rgb_image,
    caption="hammer",  # Just describe what you want!
    box_threshold=0.3,
    text_threshold=0.25
)

# boxes contains [x1, y1, x2, y2] of detected hammers
```

**Pros**:
- No training needed
- Works on any object (just change text prompt)
- Very robust

**Cons**:
- Requires ~2GB model download
- ~100-200ms inference

#### **B. Segment Anything Model (SAM)**
```python
from segment_anything import sam_model_registry, SamPredictor

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Give it a point on the hammer (from color heuristic)
predictor.set_image(rgb_image)
masks, scores, _ = predictor.predict(
    point_coords=hammer_point,  # Click on hammer
    point_labels=[1],  # Foreground
)

# masks contains pixel-perfect segmentation
```

**Pros**:
- State-of-the-art segmentation
- Very accurate
- Works on any object

**Cons**:
- Needs a starting point/box
- ~500ms inference (heavy model)

---

### **Option 2: Classical Object Detection** ⭐ Fast & Reliable

Use standard detection models, fine-tune on your data.

#### **A. YOLOv8** (Recommended)
```python
from ultralytics import YOLO

# Option 1: Pre-trained (may already detect "hammer")
model = YOLO("yolov8n.pt")  # Nano model, fast
results = model(rgb_image)

# Option 2: Fine-tune on your data
model = YOLO("yolov8n.pt")
model.train(
    data="hammer_dataset.yaml",  # Your labeled images
    epochs=50,
    imgsz=640
)

# Inference
results = model(rgb_image)
for box in results[0].boxes:
    if box.cls == 'hammer':
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
```

**Pros**:
- Very fast (10-30ms)
- Easy to fine-tune
- Production-ready

**Cons**:
- Needs labeled data for fine-tuning
- May require 50-100 annotated images

#### **B. Faster R-CNN / Mask R-CNN**
```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inference
predictions = model(rgb_tensor)
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']
```

---

### **Option 3: End-to-End RL Vision** ⭐ What You're Already Building!

**Important**: The PointNet approach you have IS a learned vision model!

```python
# Your current system (pointcloud_grasp_env.py):
class PointNetExtractor(BaseFeaturesExtractor):
    def forward(self, observations):
        # This LEARNS to recognize the hammer during training
        point_cloud = observations[:, :-7]  # Extract points
        features = self.point_net(point_cloud)  # Learned encoding
        return features
```

**How it works**:
1. PointNet learns features directly from point clouds
2. Through RL training, it learns: "these point patterns = hammer"
3. No explicit detection needed - it's implicit in the policy

**Pros**:
- No separate vision model needed
- Learns exactly what's useful for grasping
- End-to-end optimization

**Cons**:
- Requires RL training
- Less interpretable

---

### **Option 4: Semantic Segmentation**

Pixel-wise classification of the image.

#### **DeepLabV3+**
```python
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(pretrained=True)
model.eval()

# Inference
output = model(rgb_tensor)
segmentation = output['out'].argmax(1)

# Find hammer pixels
hammer_mask = (segmentation == HAMMER_CLASS)
hammer_points = np.argwhere(hammer_mask)
centroid = hammer_points.mean(axis=0)
```

---

## 🚀 Practical Integration

### **Recommended Approach: Grounding DINO + Your RL System**

Best of both worlds - use Grounding DINO for detection, RL for grasping:

```python
# detection_env.py
from groundingdino.util.inference import load_model, predict
import numpy as np

class GroundingDINOGraspEnv(PointCloudGraspEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load Grounding DINO
        self.detector = load_model("weights/groundingdino_swint_ogc.pth")

    def detect_hammer_with_dino(self, rgb_image):
        """Use Grounding DINO to find hammer."""
        boxes, scores, phrases = predict(
            model=self.detector,
            image=rgb_image,
            caption="hammer tool",
            box_threshold=0.35
        )

        if len(boxes) == 0:
            return None

        # Get best detection
        best_idx = scores.argmax()
        box = boxes[best_idx]  # [x1, y1, x2, y2]
        score = scores[best_idx]

        # Convert box to 3D position using depth
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)

        depth_value = self.depth_image[cy, cx]

        # Back-project to 3D
        K = self.camera_processor.get_camera_intrinsics(self.camera_name)
        pos_3d = self.pixel_to_3d(cx, cy, depth_value, K)

        return {
            'position': pos_3d,
            'bbox': box,
            'confidence': score
        }

    def reset(self):
        # Capture image
        rgb, depth = self._capture_rgbd()

        # Detect hammer with learned model
        detection = self.detect_hammer_with_dino(rgb)

        if detection:
            # Move arm near detected hammer
            self._approach_target(detection['position'])

            # Now use RL for fine grasping
            obs = self._get_observation()
            return obs, {'detection_confidence': detection['confidence']}
        else:
            # Fallback or retry
            return self._get_observation(), {'detection_confidence': 0.0}
```

---

## 📊 Comparison of Methods

| Method | Training Data Needed | Inference Speed | Accuracy | Generalization |
|--------|---------------------|-----------------|----------|----------------|
| **Grounding DINO** | None (zero-shot) | 100-200ms | 90%+ | Excellent |
| **YOLOv8** | 50-100 images | 10-30ms | 95%+ | Good |
| **SAM** | None | 500ms | 98%+ | Excellent |
| **PointNet (RL)** | RL training | 5ms | 80%+ | Medium |
| **Color Heuristic** | Manual tuning | 10ms | 60-70% | Poor |

---

## 🎓 Recommended Path

### **For Your Project**: Use Grounding DINO + RL

**Phase 1: Quick Validation** (This week)
```bash
# Install Grounding DINO
pip install groundingdino-py

# Download weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Test detection
python test_grounding_dino.py --image pointcloud_data/rgb.png --text "hammer"
```

**Phase 2: Integrate with RL** (Next week)
```python
# Use detection to initialize arm position
# Then RL takes over for fine control
# Best of both worlds!
```

**Why this is best**:
- ✅ No training data needed (zero-shot)
- ✅ Very robust (works on any object)
- ✅ Fast enough for RL (100ms is fine for reset)
- ✅ Complements RL (coarse detection + fine RL control)

---

## 💻 Implementation Example

Let me create a complete example for you:

```python
# grounding_dino_detector.py
import torch
import numpy as np
from PIL import Image
from groundingdino.util.inference import load_model, predict

class GroundingDINODetector:
    """Wrapper for Grounding DINO object detection."""

    def __init__(self, model_path="weights/groundingdino_swint_ogc.pth"):
        self.model = load_model(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, rgb_image, text_prompt="hammer", threshold=0.35):
        """
        Detect object using text prompt.

        Args:
            rgb_image: numpy array (H, W, 3)
            text_prompt: what to detect (e.g., "hammer", "tool", "red hammer")
            threshold: confidence threshold

        Returns:
            List of detections with boxes and scores
        """
        # Convert to PIL
        pil_image = Image.fromarray(rgb_image)

        # Run detection
        boxes, logits, phrases = predict(
            model=self.model,
            image=pil_image,
            caption=text_prompt,
            box_threshold=threshold,
            text_threshold=0.25
        )

        # Convert to list of dicts
        detections = []
        for box, score, phrase in zip(boxes, logits, phrases):
            detections.append({
                'bbox': box.tolist(),  # [x1, y1, x2, y2] normalized
                'confidence': float(score),
                'label': phrase
            })

        return detections

    def get_3d_position(self, detection, depth_image, camera_intrinsics):
        """Convert 2D detection to 3D position using depth."""
        h, w = depth_image.shape
        box = detection['bbox']

        # Denormalize box
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        # Center point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Get depth
        depth_value = depth_image[cy, cx]

        # Back-project using camera intrinsics
        K = camera_intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx_cam, cy_cam = K[0, 2], K[1, 2]

        z = depth_value
        x = (cx - cx_cam) * z / fx
        y = (cy - cy_cam) * z / fy

        return np.array([x, y, z])


# Example usage
if __name__ == "__main__":
    detector = GroundingDINODetector()

    # Load test image
    rgb = np.array(Image.open("test_image.png"))

    # Detect
    detections = detector.detect(rgb, text_prompt="hammer")

    print(f"Found {len(detections)} hammer(s):")
    for i, det in enumerate(detections):
        print(f"  {i+1}. Confidence: {det['confidence']:.2%}, Box: {det['bbox']}")
```

---

## 🎯 Summary

**Yes, you should use learned models!** Here's the recommendation:

### **Best Approach for You**:

1. **Grounding DINO for coarse detection** (zero-shot, no training)
   - Detects hammer from any viewpoint
   - Gives rough 3D position
   - Very robust

2. **Your PointNet RL for fine control** (already built!)
   - Refines the grasp
   - Learns from experience
   - Optimizes for your robot

3. **Wrist camera for final approach** (what we just added)
   - Direct visual feedback
   - Faster training
   - Higher success rate

### **Implementation Timeline**:

**Week 1**: Get Grounding DINO working standalone
**Week 2**: Integrate with your RL environment
**Week 3**: Train hybrid system
**Week 4**: Achieve 80%+ success rate!

This is **exactly how modern robot learning works** - foundation models for perception, RL for control.

Want me to create the Grounding DINO integration code?
