#!/usr/bin/env python3
"""
Grounding DINO detector for hammer recognition.

Zero-shot object detection using natural language prompts.
"""

import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path


class GroundingDINODetector:
    """Wrapper for Grounding DINO object detection."""

    def __init__(self):
        """Initialize Grounding DINO detector."""
        from groundingdino.util.inference import load_model, load_image, predict
        from huggingface_hub import hf_hub_download

        # Download config and checkpoint from HuggingFace
        print("Loading Grounding DINO model...")

        cache_dir = Path.home() / ".cache" / "groundingdino"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download config
        config_path = hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO",
            filename="GroundingDINO_SwinT_OGC.cfg.py",
            cache_dir=cache_dir
        )

        # Download checkpoint
        checkpoint_path = hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO",
            filename="groundingdino_swint_ogc.pth",
            cache_dir=cache_dir
        )

        # Load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(config_path, checkpoint_path, device=self.device)

        print(f"✓ Grounding DINO initialized on {self.device}")

    def detect(self, rgb_image, text_prompt="hammer", box_threshold=0.35, text_threshold=0.25):
        """
        Detect objects using text prompt.

        Args:
            rgb_image: numpy array (H, W, 3) uint8
            text_prompt: what to detect (e.g., "hammer", "tool", "red hammer")
            box_threshold: confidence threshold for detection
            text_threshold: text matching threshold

        Returns:
            List of detections with boxes, scores, and labels
        """
        from groundingdino.util.inference import predict, load_image
        from groundingdino.util import box_ops
        import tempfile
        import os

        # Convert to PIL Image if needed
        if isinstance(rgb_image, np.ndarray):
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = rgb_image

        # Save to temp file and load with groundingdino's loader
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            pil_image.save(temp_path)

        try:
            # Load image using groundingdino's helper (converts to tensor)
            image_source, image_tensor = load_image(temp_path)

            # Run detection - returns boxes in normalized coords [0,1]
            boxes, confidences, labels = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
        finally:
            # Clean up temp file
            os.unlink(temp_path)

        # Convert to list of dicts
        results = []
        h, w = rgb_image.shape[:2]

        for i in range(len(boxes)):
            box = boxes[i]  # normalized coords [cx, cy, w, h]

            # Convert from center format to corner format and scale to pixels
            cx, cy, bw, bh = box
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            # Clamp to image bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))

            results.append({
                'bbox': [x1, y1, x2, y2],
                'bbox_normalized': [x1/w, y1/h, x2/w, y2/h],
                'confidence': float(confidences[i]),
                'label': str(labels[i]) if labels[i] else text_prompt,
                'center': [(x1+x2)//2, (y1+y2)//2]
            })

        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results

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
    """Test Grounding DINO on hammer image."""
    import mujoco
    from camera_utils import CameraProcessor

    print("="*60)
    print("Testing Grounding DINO on Hammer Detection")
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
    print("\n3. Initializing Grounding DINO...")
    detector = GroundingDINODetector()

    # Detect hammer
    print("\n4. Detecting 'hammer' with zero-shot recognition...")
    detections = detector.detect(rgb, text_prompt="hammer", box_threshold=0.25)

    print(f"\n5. Results:")
    print(f"   Found {len(detections)} detection(s)")

    if len(detections) > 0:
        for i, det in enumerate(detections[:3]):  # Show top 3
            print(f"\n   Detection #{i+1}:")
            print(f"     Confidence: {det['confidence']:.2%}")
            print(f"     Bounding box: {det['bbox']}")
            print(f"     Center (pixels): {det['center']}")

        # Get 3D position
        best_detection = detections[0]
        processor = CameraProcessor(model, width=640, height=480)
        K = processor.get_camera_intrinsics("track_front")

        pos_3d = detector.get_3d_position(best_detection, depth, K)
        pos_world = processor.camera_to_world_frame(pos_3d.reshape(1, 3), data, "track_front")[0]

        print(f"\n   3D Position (best detection):")
        print(f"     Camera frame: [{pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f}] m")
        print(f"     World frame:  [{pos_world[0]:.3f}, {pos_world[1]:.3f}, {pos_world[2]:.3f}] m")

        # Ground truth comparison
        hammer_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hammer")
        gt_pos = data.xpos[hammer_body_id]
        error = np.linalg.norm(pos_world - gt_pos)

        print(f"\n   Ground Truth: [{gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f}] m")
        print(f"   Detection Error: {error:.3f} m ({error*100:.1f} cm)")

        # Visualize
        print("\n6. Saving visualization...")
        output_dir = Path(__file__).parent / "grounding_dino_results"
        output_dir.mkdir(exist_ok=True)

        vis_path = output_dir / "hammer_detection.png"
        detector.visualize_detections(rgb, detections, save_path=str(vis_path))

        print(f"\n{'='*60}")
        print("✓ Test Complete!")
        print(f"{'='*60}")
        print(f"Detection confidence: {best_detection['confidence']:.1%}")
        print(f"Position accuracy: ±{error*100:.1f} cm")
        print(f"Visualization: {vis_path}")

    else:
        print("\n✗ No hammer detected!")
        print("Try adjusting:")
        print("  - box_threshold (lower = more detections)")
        print("  - text_prompt (try 'tool', 'hammer tool', 'claw hammer')")


if __name__ == "__main__":
    test_on_hammer()
