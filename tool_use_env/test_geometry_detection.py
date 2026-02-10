#!/usr/bin/env python3
"""
Test Grounding DINO with full geometry analysis on your images.
"""

import numpy as np
import cv2
from grounding_dino_detector import GroundingDINODetector

print("="*60)
print("Testing Grounding DINO Geometry Detection")
print("="*60)

# Initialize detector
print("\n1. Initializing Grounding DINO...")
detector = GroundingDINODetector()

# Load RGB image
print("\n2. Loading images...")
rgb = cv2.imread("pointcloud_data/rgb.png")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
print(f"   RGB shape: {rgb.shape}")

# Load depth image
depth_img = cv2.imread("pointcloud_data/depth.png", cv2.IMREAD_UNCHANGED)
print(f"   Depth shape: {depth_img.shape}")
print(f"   Depth range: {depth_img.min()} - {depth_img.max()}")

# Handle RGBA depth images - take first channel
if len(depth_img.shape) == 3:
    depth_img = depth_img[:, :, 0]  # Use first channel

# Convert depth to meters (adjust based on your encoding)
# If stored as 16-bit with mm, divide by 1000
# If stored as 8-bit normalized, need to scale differently
if depth_img.dtype == np.uint16:
    depth = depth_img.astype(np.float32) / 1000.0  # mm to m
elif depth_img.max() <= 255:
    # Normalized to 0-255, estimate actual range
    # Assuming scene depth is 0.5m to 2m
    depth = depth_img.astype(np.float32) / 255.0 * 1.5 + 0.5
else:
    depth = depth_img.astype(np.float32)

print(f"   Depth in meters: {depth.min():.3f} - {depth.max():.3f} m")

# Try different prompts
prompts = [
    "hammer",
    "robot arm",
    "metal tool",
    "claw hammer"
]

print("\n3. Testing different prompts...")
for prompt in prompts:
    print(f"\n   Prompt: '{prompt}'")
    detections = detector.detect(rgb, text_prompt=prompt, box_threshold=0.25)
    print(f"   Found: {len(detections)} detection(s)")
    if len(detections) > 0:
        best = detections[0]
        print(f"   Best: {best['confidence']:.1%} at {best['bbox']}")

# Detect hammer with best settings
print("\n4. Detecting hammer with detailed geometry...")
detections = detector.detect(
    rgb,
    text_prompt="hammer",
    box_threshold=0.30  # Adjust threshold
)

print(f"\n   Found {len(detections)} detection(s)")

# Analyze each detection
for i, det in enumerate(detections):
    print(f"\n   --- Detection {i+1} ---")
    print(f"   Confidence: {det['confidence']:.2%}")
    print(f"   Bounding Box: {det['bbox']}")

    x1, y1, x2, y2 = det['bbox']
    cx, cy = det['center']

    # Size in pixels
    width_px = x2 - x1
    height_px = y2 - y1
    print(f"   Size: {width_px} x {height_px} pixels")

    # Get depth at center
    if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
        depth_center = depth[cy, cx]

        # Get depth statistics in bounding box
        depth_roi = depth[y1:y2, x1:x2]
        depth_mean = depth_roi.mean()
        depth_std = depth_roi.std()

        print(f"   Depth (center): {depth_center:.3f} m")
        print(f"   Depth (mean): {depth_mean:.3f} ± {depth_std:.3f} m")

        # Estimate physical size (rough estimate)
        # Assuming FOV ~60 degrees for typical camera
        # width_m = (width_px / image_width) * (2 * depth * tan(FOV/2))
        # For simplicity: width_m ≈ (width_px / focal_length) * depth

        # Typical focal length for 640x480 camera
        focal_length_px = 525.0  # adjust if you know your camera's focal length

        width_m = (width_px / focal_length_px) * depth_center
        height_m = (height_px / focal_length_px) * depth_center

        print(f"   Estimated size: {width_m*100:.1f} x {height_m*100:.1f} cm")

        # Determine if this is the actual hammer (not robot arm)
        # The hammer should be:
        # - Smaller than robot arms
        # - Further away (higher Y in image = further in this perspective)
        # - More horizontal aspect ratio
        aspect_ratio = width_px / max(height_px, 1)

        print(f"   Aspect ratio: {aspect_ratio:.2f}")
        print(f"   Position in image: center={cx}, top={y1}")

        if y1 < rgb.shape[0] // 3 and aspect_ratio > 2.0:
            print(f"   *** Likely the ACTUAL HAMMER ***")

# Save visualization
print("\n5. Saving visualization...")
vis_img = detector.visualize_detections(rgb, detections, save_path="geometry_detection_result.png")

print("\n" + "="*60)
print("✓ Detection Complete!")
print("="*60)
print(f"Results saved to: geometry_detection_result.png")
