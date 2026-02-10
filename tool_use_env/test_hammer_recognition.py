#!/usr/bin/env python3
"""
Test and demonstrate hammer recognition methods.

This script shows all available methods for detecting the hammer
and measuring its distance/position.
"""

import numpy as np
import mujoco
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def method_1_physics_based(model, data):
    """
    Method 1: Direct physics-based position (Ground Truth)

    How it works:
    - MuJoCo tracks exact position of every body
    - Query hammer body ID and get its position directly
    - This is "cheating" but useful for training/debugging

    Pros: Perfect accuracy, instant, no computation
    Cons: Not available on real robots
    """
    print("\n" + "="*60)
    print("METHOD 1: Physics-Based Position (Ground Truth)")
    print("="*60)

    # Get hammer body ID
    hammer_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hammer")

    if hammer_body_id < 0:
        print("ERROR: Hammer body not found!")
        return None

    # Get exact position from physics engine
    hammer_pos = data.xpos[hammer_body_id].copy()
    hammer_vel = data.cvel[hammer_body_id, :3].copy()  # Linear velocity
    hammer_quat = data.xquat[hammer_body_id].copy()  # Orientation

    print(f"How it works:")
    print(f"  - Query MuJoCo's physics state directly")
    print(f"  - Body ID: {hammer_body_id}")
    print(f"  - No vision, no processing required")

    print(f"\nResults:")
    print(f"  Position: [{hammer_pos[0]:.4f}, {hammer_pos[1]:.4f}, {hammer_pos[2]:.4f}] m")
    print(f"  Velocity: [{hammer_vel[0]:.4f}, {hammer_vel[1]:.4f}, {hammer_vel[2]:.4f}] m/s")
    print(f"  Orientation (quat): [{hammer_quat[0]:.3f}, {hammer_quat[1]:.3f}, {hammer_quat[2]:.3f}, {hammer_quat[3]:.3f}]")

    print(f"\nPros: ✓ Perfect accuracy, ✓ Instant, ✓ No computation")
    print(f"Cons: ✗ Not available on real robots (simulation only)")
    print(f"Use case: Training, debugging, reward computation")

    return {
        'method': 'physics',
        'position': hammer_pos,
        'velocity': hammer_vel,
        'orientation': hammer_quat,
        'confidence': 1.0
    }


def method_2_color_segmentation(model, data, camera_name='track_front'):
    """
    Method 2: Color-Based Segmentation

    How it works:
    1. Capture RGB image from camera
    2. Filter pixels matching hammer color (brown/orange)
    3. Cluster filtered pixels
    4. Compute centroid of largest cluster
    5. Use depth to get 3D position

    Pros: Fast, simple, works without training
    Cons: Sensitive to lighting, only works for known colors
    """
    print("\n" + "="*60)
    print("METHOD 2: Color-Based Segmentation")
    print("="*60)

    from camera_utils import CameraProcessor

    # Setup
    width, height = 640, 480
    processor = CameraProcessor(model, width=width, height=height)
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Capture RGB
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    # Capture depth
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    print(f"How it works:")
    print(f"  1. Capture RGB image ({width}x{height})")
    print(f"  2. Define hammer color range (brown/orange)")
    print(f"  3. Filter pixels matching color")
    print(f"  4. Cluster and find largest group")
    print(f"  5. Use depth to compute 3D position")

    # Define hammer color range (HSV for better color matching)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)

    # Brown/orange color range in HSV
    lower_brown = np.array([10, 50, 50])   # Hue, Saturation, Value
    upper_brown = np.array([30, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("\n✗ No hammer pixels found!")
        print(f"  Try adjusting color range or check lighting")
        return None

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)

    if M['m00'] == 0:
        print("\n✗ Invalid contour!")
        return None

    # Centroid in image coordinates
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Get depth at centroid
    depth_value = depth[cy, cx]

    # Convert to 3D using camera intrinsics
    K = processor.get_camera_intrinsics(camera_name)
    fx, fy = K[0, 0], K[1, 1]
    cx_cam, cy_cam = K[0, 2], K[1, 2]

    # Back-project to 3D (camera frame)
    z = depth_value
    x = (cx - cx_cam) * z / fx
    y = (cy - cy_cam) * z / fy
    pos_camera = np.array([x, y, z])

    # Transform to world frame
    pos_world = processor.camera_to_world_frame(pos_camera.reshape(1, 3), data, camera_name)[0]

    print(f"\nProcessing:")
    print(f"  Hammer pixels found: {len(largest_contour)} px")
    print(f"  Image centroid: ({cx}, {cy}) px")
    print(f"  Depth at centroid: {depth_value:.3f} m")

    print(f"\nResults:")
    print(f"  Position (camera frame): [{pos_camera[0]:.4f}, {pos_camera[1]:.4f}, {pos_camera[2]:.4f}] m")
    print(f"  Position (world frame): [{pos_world[0]:.4f}, {pos_world[1]:.4f}, {pos_world[2]:.4f}] m")

    print(f"\nPros: ✓ Fast (~10ms), ✓ No training needed, ✓ Simple")
    print(f"Cons: ✗ Lighting sensitive, ✗ Only works for known colors")
    print(f"Use case: Quick detection, known objects, controlled lighting")

    return {
        'method': 'color_segmentation',
        'position': pos_world,
        'confidence': min(len(largest_contour) / 1000.0, 1.0),  # Based on pixel count
        'pixel_count': len(largest_contour)
    }


def method_3_point_cloud_clustering(model, data, camera_name='track_front'):
    """
    Method 3: Point Cloud Clustering

    How it works:
    1. Generate full point cloud from RGB-D
    2. Filter by color (brown/orange points)
    3. Apply DBSCAN clustering
    4. Find largest cluster
    5. Compute cluster centroid

    Pros: More robust than 2D, gets 3D shape
    Cons: Slower, needs tuning
    """
    print("\n" + "="*60)
    print("METHOD 3: Point Cloud Clustering")
    print("="*60)

    from camera_utils import CameraProcessor
    from sklearn.cluster import DBSCAN

    # Setup
    width, height = 640, 480
    processor = CameraProcessor(model, width=width, height=height)
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Capture RGB-D
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    print(f"How it works:")
    print(f"  1. Generate 3D point cloud from RGB-D")
    print(f"  2. Filter points by hammer color")
    print(f"  3. Apply DBSCAN spatial clustering")
    print(f"  4. Find largest cluster (the hammer)")
    print(f"  5. Compute 3D centroid and bounding box")

    # Generate point cloud
    points, colors = processor.rgbd_to_pointcloud(rgb, depth, camera_name,
                                                   min_depth=0.1, max_depth=3.0)

    print(f"\nProcessing:")
    print(f"  Total points: {len(points):,}")

    # Filter by color (brown/orange)
    color_mask = np.all((colors >= [80, 50, 20]) & (colors <= [255, 200, 150]), axis=1)
    hammer_points = points[color_mask]

    print(f"  Hammer-colored points: {len(hammer_points):,}")

    if len(hammer_points) < 100:
        print("\n✗ Too few hammer points found!")
        return None

    # Cluster
    clustering = DBSCAN(eps=0.05, min_samples=10).fit(hammer_points)
    labels = clustering.labels_

    # Find largest cluster (ignore noise label=-1)
    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) == 0:
        print("\n✗ No clusters found!")
        return None

    cluster_sizes = {label: (labels == label).sum() for label in unique_labels}
    largest_label = max(cluster_sizes, key=cluster_sizes.get)

    # Get largest cluster points
    cluster_points = hammer_points[labels == largest_label]

    # Compute statistics
    centroid = cluster_points.mean(axis=0)
    min_bound = cluster_points.min(axis=0)
    max_bound = cluster_points.max(axis=0)
    bbox_size = max_bound - min_bound

    print(f"  Clusters found: {len(unique_labels)}")
    print(f"  Largest cluster: {len(cluster_points):,} points")

    print(f"\nResults:")
    print(f"  Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}] m")
    print(f"  Bounding box: [{bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f}] m")
    print(f"  Min bound: [{min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}] m")
    print(f"  Max bound: [{max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f}] m")

    print(f"\nPros: ✓ Robust to viewpoint, ✓ Gets 3D shape, ✓ Multiple metrics")
    print(f"Cons: ✗ Slower (~50-100ms), ✗ Needs parameter tuning")
    print(f"Use case: Precise localization, shape analysis, grasp planning")

    return {
        'method': 'point_cloud_clustering',
        'position': centroid,
        'bounding_box': bbox_size,
        'min_bound': min_bound,
        'max_bound': max_bound,
        'confidence': min(len(cluster_points) / 5000.0, 1.0),
        'num_points': len(cluster_points)
    }


def method_4_distance_to_gripper(model, data, hammer_result):
    """
    Method 4: Compute Distance to Gripper

    How it works:
    - Get gripper position from robot state
    - Compute Euclidean distance to hammer
    - This is what RL uses for reward shaping

    Pros: Simple, direct, used in RL reward
    Cons: Requires knowing both positions
    """
    print("\n" + "="*60)
    print("METHOD 4: Distance Measurement")
    print("="*60)

    # Get gripper position
    right_hand_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    if right_hand_site_id < 0:
        print("ERROR: Right palm site not found!")
        return None

    gripper_pos = data.site_xpos[right_hand_site_id].copy()
    hammer_pos = hammer_result['position']

    # Compute distance
    distance = np.linalg.norm(gripper_pos - hammer_pos)

    # Compute direction
    direction = hammer_pos - gripper_pos
    direction_norm = direction / (distance + 1e-8)

    print(f"How it works:")
    print(f"  - Query gripper position from robot state")
    print(f"  - Compute Euclidean distance: ||hammer - gripper||")
    print(f"  - Used in RL reward: reward = -distance * weight")

    print(f"\nPositions:")
    print(f"  Gripper: [{gripper_pos[0]:.4f}, {gripper_pos[1]:.4f}, {gripper_pos[2]:.4f}] m")
    print(f"  Hammer:  [{hammer_pos[0]:.4f}, {hammer_pos[1]:.4f}, {hammer_pos[2]:.4f}] m")

    print(f"\nResults:")
    print(f"  Distance: {distance:.4f} m ({distance*100:.2f} cm)")
    print(f"  Direction (unit vector): [{direction_norm[0]:.3f}, {direction_norm[1]:.3f}, {direction_norm[2]:.3f}]")

    # Categorize distance
    if distance < 0.05:
        status = "✓ In grasp range (< 5cm)"
    elif distance < 0.15:
        status = "○ Close (5-15cm)"
    elif distance < 0.30:
        status = "△ Approaching (15-30cm)"
    else:
        status = "✗ Far (> 30cm)"

    print(f"  Status: {status}")

    print(f"\nPros: ✓ Simple, ✓ Fast, ✓ Direct feedback for RL")
    print(f"Cons: ✗ Requires knowing both positions")
    print(f"Use case: RL reward shaping, approach control")

    return {
        'gripper_position': gripper_pos,
        'hammer_position': hammer_pos,
        'distance': distance,
        'direction': direction_norm
    }


def compare_methods(results):
    """Compare all methods and show which to use when."""
    print("\n" + "="*60)
    print("COMPARISON & RECOMMENDATIONS")
    print("="*60)

    if 'physics' in results and 'color_segmentation' in results:
        physics_pos = results['physics']['position']
        color_pos = results['color_segmentation']['position']
        error = np.linalg.norm(physics_pos - color_pos)

        print(f"\nAccuracy Check (vs Ground Truth):")
        print(f"  Color Segmentation Error: {error:.4f} m ({error*100:.2f} cm)")

    if 'physics' in results and 'point_cloud_clustering' in results:
        physics_pos = results['physics']['position']
        pc_pos = results['point_cloud_clustering']['position']
        error = np.linalg.norm(physics_pos - pc_pos)

        print(f"  Point Cloud Error: {error:.4f} m ({error*100:.2f} cm)")

    print(f"\n📊 Method Comparison:")
    print(f"{'Method':<25} {'Speed':<15} {'Accuracy':<15} {'Best For'}")
    print(f"{'-'*80}")
    print(f"{'Physics (Ground Truth)':<25} {'Instant':<15} {'Perfect':<15} {'Simulation/Training'}")
    print(f"{'Color Segmentation':<25} {'~10ms':<15} {'±2-5cm':<15} {'Fast detection'}")
    print(f"{'Point Cloud Clustering':<25} {'~50-100ms':<15} {'±1-2cm':<15} {'Precise localization'}")

    print(f"\n✅ Recommendations:")
    print(f"  For RL Training: Use Physics (Method 1) - instant, perfect")
    print(f"  For Real Robot: Use Point Cloud (Method 3) - most robust")
    print(f"  For Fast Detection: Use Color (Method 2) - good speed/accuracy tradeoff")
    print(f"  For Reward: Use Distance (Method 4) - direct feedback")


def main():
    """Run all hammer recognition tests."""
    print("="*60)
    print("HAMMER RECOGNITION TEST SUITE")
    print("="*60)
    print("\nThis demonstrates all methods for detecting the hammer")
    print("and measuring distances in the simulation.\n")

    # Load scene
    scene_path = Path(__file__).parent / "hammer_grasp_rgbd_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Let hammer settle
    print("Initializing scene...")
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Run all methods
    results = {}

    # Method 1: Physics-based (ground truth)
    results['physics'] = method_1_physics_based(model, data)

    # Method 2: Color segmentation
    results['color_segmentation'] = method_2_color_segmentation(model, data)

    # Method 3: Point cloud clustering
    results['point_cloud_clustering'] = method_3_point_cloud_clustering(model, data)

    # Method 4: Distance measurement
    if results['physics']:
        results['distance'] = method_4_distance_to_gripper(model, data, results['physics'])

    # Compare and recommend
    compare_methods(results)

    print("\n" + "="*60)
    print("Test complete! See results above.")
    print("="*60)


if __name__ == "__main__":
    main()
