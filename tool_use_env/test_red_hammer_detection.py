#!/usr/bin/env python3
"""
Test color-based detection with the new bright red hammer.
"""

import numpy as np
import cv2
import mujoco
from pathlib import Path
from camera_utils import CameraProcessor


def detect_red_hammer(rgb_image, depth_image, camera_processor, camera_name, data):
    """
    Detect bright red hammer using color segmentation.

    Returns:
        dict with detection info or None if not found
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Red color range in HSV
    # Red wraps around in HSV (0-10 and 170-180)
    # Lower red range (0-10 degrees)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    # Upper red range (170-180 degrees)
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Get largest contour (should be the hammer)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 100:  # Filter out noise
        return None

    # Get centroid
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Get depth at centroid
    depth_value = depth_image[cy, cx]

    # Back-project to 3D camera frame
    K = camera_processor.get_camera_intrinsics(camera_name)
    fx, fy = K[0, 0], K[1, 1]
    cx_cam, cy_cam = K[0, 2], K[1, 2]

    z = depth_value
    x_3d = (cx - cx_cam) * z / fx
    y_3d = (cy - cy_cam) * z / fy

    pos_camera = np.array([x_3d, y_3d, z])

    # Convert to world frame
    pos_world = camera_processor.camera_to_world_frame(
        pos_camera.reshape(1, 3), data, camera_name
    )[0]

    return {
        'position_world': pos_world,
        'position_camera': pos_camera,
        'centroid_2d': (cx, cy),
        'bbox': (x, y, w, h),
        'area': area,
        'mask': red_mask,
        'num_pixels': np.count_nonzero(red_mask)
    }


def visualize_detection(rgb_image, detection):
    """Create visualization of detection."""
    vis = rgb_image.copy()

    if detection is None:
        return vis

    # Draw bounding box
    x, y, w, h = detection['bbox']
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw centroid
    cx, cy = detection['centroid_2d']
    cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)

    # Add text
    pos = detection['position_world']
    text = f"Hammer: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]m"
    cv2.putText(vis, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add pixel count
    text2 = f"Pixels: {detection['num_pixels']}"
    cv2.putText(vis, text2, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return vis


def main():
    print("\n" + "="*60)
    print("Testing Red Hammer Detection")
    print("="*60 + "\n")

    # Load scene
    scene_path = Path(__file__).parent / "hammer_grasp_rgbd_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Let hammer settle
    print("1. Initializing scene...")
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Get ground truth
    hammer_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hammer")
    gt_pos = data.xpos[hammer_body_id].copy()
    print(f"   Ground truth position: [{gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f}] m\n")

    # Capture RGB-D
    print("2. Capturing RGB-D image...")
    camera_name = "track_front"
    renderer = mujoco.Renderer(model, height=480, width=640)

    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    print(f"   RGB shape: {rgb.shape}")
    print(f"   Depth range: {depth.min():.3f} - {depth.max():.3f} m\n")

    # Initialize camera processor
    camera_processor = CameraProcessor(model, width=640, height=480)

    # Detect hammer
    print("3. Detecting red hammer...")
    detection = detect_red_hammer(rgb, depth, camera_processor, camera_name, data)

    if detection is not None:
        print(f"   ✓ Hammer detected!")
        print(f"   Position (world): [{detection['position_world'][0]:.3f}, "
              f"{detection['position_world'][1]:.3f}, {detection['position_world'][2]:.3f}] m")
        print(f"   Bounding box: {detection['bbox']}")
        print(f"   Area: {detection['area']:.0f} pixels")
        print(f"   Red pixels detected: {detection['num_pixels']}")

        # Calculate error
        error = np.linalg.norm(detection['position_world'] - gt_pos)
        print(f"\n   Ground truth: [{gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f}] m")
        print(f"   Detection error: {error:.3f} m ({error*100:.1f} cm)")

        if error < 0.05:
            print(f"   ✓ Excellent accuracy! (< 5cm)")
        elif error < 0.10:
            print(f"   ✓ Good accuracy (< 10cm)")
        else:
            print(f"   ⚠ Moderate accuracy")

    else:
        print("   ✗ No red hammer detected!")

    # Save visualizations
    print("\n4. Saving visualizations...")
    output_dir = Path(__file__).parent / "red_hammer_detection"
    output_dir.mkdir(exist_ok=True)

    # Save original RGB
    cv2.imwrite(str(output_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Save mask
    if detection is not None:
        cv2.imwrite(str(output_dir / "red_mask.png"), detection['mask'])

        # Save visualization
        vis = visualize_detection(rgb, detection)
        cv2.imwrite(str(output_dir / "detection.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # Create side-by-side comparison
        mask_colored = cv2.cvtColor(detection['mask'], cv2.COLOR_GRAY2BGR)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        comparison = np.hstack([vis_bgr, mask_colored])
        cv2.imwrite(str(output_dir / "comparison.png"), comparison)

    print(f"   Saved to: {output_dir}/")
    print(f"     - rgb.png (original)")
    if detection is not None:
        print(f"     - red_mask.png (segmentation mask)")
        print(f"     - detection.png (with bounding box)")
        print(f"     - comparison.png (side-by-side)")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

    # Compare with old orange detection
    if detection is not None:
        print(f"\n💡 Improvement over orange hammer:")
        print(f"   Old detection: ~800 pixels (orange)")
        print(f"   New detection: {detection['num_pixels']} pixels (bright red)")
        improvement = (detection['num_pixels'] / 800 - 1) * 100
        if improvement > 0:
            print(f"   {improvement:.0f}% more pixels detected! Much easier to see!")


if __name__ == "__main__":
    main()
