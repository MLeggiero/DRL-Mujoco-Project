#!/usr/bin/env python3
"""
Simple example: Use Grounding DINO to detect object and get 3D geometry
"""

import numpy as np
import cv2
from pathlib import Path
from grounding_dino_detector import GroundingDINODetector


def example_from_image_file():
    """Detect object from a saved image."""

    # Initialize detector
    print("Initializing Grounding DINO...")
    detector = GroundingDINODetector()

    # Load your image
    image_path = "pointcloud_data/rgb.png"  # Change to your image path
    rgb = cv2.imread(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Detect object using natural language
    # Try different prompts: "hammer", "tool", "red object", "metal tool", etc.
    print("\nDetecting object...")
    detections = detector.detect(
        rgb,
        text_prompt="hammer",  # Change this to detect different objects!
        box_threshold=0.25,    # Lower = more detections, higher = fewer but more confident
        text_threshold=0.25
    )

    # Print results
    print(f"\nFound {len(detections)} detection(s):")
    for i, det in enumerate(detections):
        print(f"\nDetection #{i+1}:")
        print(f"  Confidence: {det['confidence']:.2%}")
        print(f"  Bounding Box: {det['bbox']} (x1, y1, x2, y2 in pixels)")
        print(f"  Center: {det['center']} (x, y in pixels)")

        # Calculate object size from bounding box
        x1, y1, x2, y2 = det['bbox']
        width_px = x2 - x1
        height_px = y2 - y1
        print(f"  Size: {width_px} x {height_px} pixels")

    # Visualize
    if len(detections) > 0:
        output_path = "grounding_dino_detection.png"
        detector.visualize_detections(rgb, detections, save_path=output_path)
        print(f"\n✓ Saved visualization to {output_path}")

    return detections


def example_with_depth():
    """Detect object and get 3D position using depth."""

    detector = GroundingDINODetector()

    # Load RGB and depth images
    rgb_path = "pointcloud_data/rgb.png"
    depth_path = "pointcloud_data/depth.png"

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Load depth (depends on your format)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # If depth is stored as PNG, you may need to convert it
    # Assuming depth is in millimeters, convert to meters
    depth = depth_img.astype(np.float32) / 1000.0

    # Detect object
    detections = detector.detect(rgb, text_prompt="hammer")

    if len(detections) > 0:
        best = detections[0]

        # Define camera intrinsics (example values - adjust for your camera!)
        # For a 640x480 image with ~60 degree FOV:
        fx = fy = 525.0  # focal length in pixels
        cx, cy = 320.0, 240.0  # principal point (image center)

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # Get 3D position
        pos_3d = detector.get_3d_position(best, depth, K)

        print(f"\n3D Position in camera frame:")
        print(f"  X: {pos_3d[0]:.3f} m (right)")
        print(f"  Y: {pos_3d[1]:.3f} m (down)")
        print(f"  Z: {pos_3d[2]:.3f} m (forward)")

        # Get depth at object center
        cx, cy = best['center']
        depth_at_center = depth[cy, cx]
        print(f"\nDepth at object center: {depth_at_center:.3f} m")

    return detections


def example_detect_multiple_objects():
    """Detect multiple different objects in one image."""

    detector = GroundingDINODetector()

    image_path = "pointcloud_data/rgb.png"
    rgb = cv2.imread(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # You can detect multiple things by combining prompts
    prompts = [
        "hammer",
        "table",
        "metal tool",
        "wooden handle"
    ]

    all_detections = {}

    for prompt in prompts:
        print(f"\nSearching for: '{prompt}'")
        detections = detector.detect(rgb, text_prompt=prompt, box_threshold=0.3)

        if len(detections) > 0:
            all_detections[prompt] = detections
            print(f"  ✓ Found {len(detections)} match(es)")
        else:
            print(f"  ✗ Not found")

    return all_detections


if __name__ == "__main__":
    print("="*60)
    print("Grounding DINO Detection Examples")
    print("="*60)

    # Choose which example to run:

    # Example 1: Basic detection from image
    print("\n--- Example 1: Basic Detection ---")
    example_from_image_file()

    # Example 2: Detection with depth (3D geometry)
    # print("\n--- Example 2: Detection with Depth ---")
    # example_with_depth()

    # Example 3: Detect multiple objects
    # print("\n--- Example 3: Multiple Objects ---")
    # example_detect_multiple_objects()
