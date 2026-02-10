#!/usr/bin/env python3
"""
YOLOv8 detector for hammer recognition.

Uses pre-trained YOLO model for object detection.
"""

import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO


class YOLODetector:
    """Wrapper for YOLOv8 object detection."""

    def __init__(self, model_size='n'):
        """
        Initialize YOLO detector.

        Args:
            model_size: Model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
                       Smaller = faster but less accurate
        """
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')  # Auto-downloads if needed
        print(f"✓ YOLOv8{model_size} loaded")

        # COCO class names (YOLOv8 is trained on COCO dataset)
        self.class_names = self.model.names

    def detect(self, rgb_image, conf_threshold=0.25, target_classes=None):
        """
        Detect objects in image.

        Args:
            rgb_image: numpy array (H, W, 3) uint8
            conf_threshold: Confidence threshold
            target_classes: List of class names to detect (None = all classes)
                          e.g., ['hammer', 'tool', 'scissors']

        Returns:
            List of detections with boxes, scores, and labels
        """
        # Run inference
        results = self.model(rgb_image, conf=conf_threshold, verbose=False)

        # Parse results
        detections = []
        h, w = rgb_image.shape[:2]

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names[class_id]

                # Filter by target classes if specified
                if target_classes and class_name not in target_classes:
                    continue

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'bbox_normalized': [x1/w, y1/h, x2/w, y2/h],
                    'confidence': confidence,
                    'label': class_name,
                    'class_id': class_id,
                    'center': [int((x1+x2)/2), int((y1+y2)/2)]
                })

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    def get_3d_position(self, detection, depth_image, camera_intrinsics):
        """
        Convert 2D detection to 3D position using depth.

        Args:
            detection: Detection dict from detect()
            depth_image: Depth map (H, W) in meters
            camera_intrinsics: 3x3 camera matrix K

        Returns:
            3D position [x, y, z] in camera frame
        """
        # Get center of bounding box
        cx, cy = detection['center']

        # Get depth at center
        depth_value = depth_image[cy, cx]

        # Back-project to 3D
        K = camera_intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx_cam, cy_cam = K[0, 2], K[1, 2]

        z = depth_value
        x = (cx - cx_cam) * z / fx
        y = (cy - cy_cam) * z / fy

        return np.array([x, y, z])

    def visualize_detections(self, rgb_image, detections, save_path=None):
        """
        Visualize detections on image.

        Args:
            rgb_image: Input RGB image
            detections: List of detections from detect()
            save_path: Optional path to save visualization

        Returns:
            Annotated image
        """
        img = rgb_image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['label']

            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            text = f"{label}: {conf:.2%}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"✓ Visualization saved to {save_path}")

        return img


def test_on_hammer():
    """Test YOLO on hammer image."""
    import mujoco
    from camera_utils import CameraProcessor

    print("="*60)
    print("Testing YOLOv8 on Hammer Detection")
    print("="*60)

    # Load scene
    scene_path = Path(__file__).parent / "hammer_grasp_rgbd_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Let hammer settle
    print("\n1. Initializing scene...")
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Capture RGB-D
    print("2. Capturing image from head camera...")
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="track_front")
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="track_front")
    depth = renderer.render()
    renderer.disable_depth_rendering()

    # Initialize detector
    print("\n3. Initializing YOLOv8...")
    detector = YOLODetector(model_size='n')  # Nano model (fastest)

    # Detect all objects first
    print("\n4. Detecting all objects in scene...")
    all_detections = detector.detect(rgb, conf_threshold=0.25)

    print(f"   Found {len(all_detections)} object(s) total:")
    for det in all_detections:
        print(f"     - {det['label']}: {det['confidence']:.1%}")

    # Try to find hammer or similar tools
    print("\n5. Looking for tool-related objects...")
    tool_classes = ['hammer', 'scissors', 'knife', 'spoon', 'fork', 'bottle']
    tool_detections = detector.detect(rgb, target_classes=tool_classes, conf_threshold=0.1)

    if len(tool_detections) > 0:
        print(f"   Found {len(tool_detections)} tool-like object(s)!")

        best_detection = tool_detections[0]

        # Get 3D position
        processor = CameraProcessor(model, width=640, height=480)
        K = processor.get_camera_intrinsics("track_front")

        pos_3d = detector.get_3d_position(best_detection, depth, K)
        pos_world = processor.camera_to_world_frame(pos_3d.reshape(1, 3), data, "track_front")[0]

        print(f"\n   Best Detection:")
        print(f"     Label: {best_detection['label']}")
        print(f"     Confidence: {best_detection['confidence']:.2%}")
        print(f"     3D Position (world): [{pos_world[0]:.3f}, {pos_world[1]:.3f}, {pos_world[2]:.3f}] m")

        # Ground truth comparison
        hammer_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hammer")
        gt_pos = data.xpos[hammer_body_id]
        error = np.linalg.norm(pos_world - gt_pos)

        print(f"     Ground Truth: [{gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f}] m")
        print(f"     Error: {error:.3f} m ({error*100:.1f} cm)")

    else:
        print("   ✗ No tool-like objects detected")
        print("   Note: YOLOv8 is pre-trained on COCO dataset")
        print("   Available classes: person, bicycle, car, etc.")
        print("\n   For best results, you can:")
        print("     1. Fine-tune YOLO on hammer images (50-100 labeled images)")
        print("     2. Use color/geometry-based detection (already working)")
        print("     3. Use RL with physics-based ground truth (recommended for training)")

    # Visualize all detections
    print("\n6. Saving visualization...")
    output_dir = Path(__file__).parent / "yolo_results"
    output_dir.mkdir(exist_ok=True)

    vis_path = output_dir / "detections.png"
    detector.visualize_detections(rgb, all_detections, save_path=str(vis_path))

    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}")
    print(f"Detections visualized: {vis_path}")

    print("\n💡 Recommendation:")
    print("   For RL training: Use physics-based position (Method 1)")
    print("   For real robot: Fine-tune YOLO on hammer images OR use color segmentation")


if __name__ == "__main__":
    test_on_hammer()
