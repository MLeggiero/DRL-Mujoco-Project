#!/usr/bin/env python3
"""
Generate point cloud from robot's head camera viewing the hammer.

This script:
1. Loads the MuJoCo scene with the robot and hammer
2. Captures RGB-D from the robot's head camera
3. Generates a point cloud
4. Saves it for visualization and grasp detection

Usage:
    python generate_pointcloud.py --camera track_front --visualize
"""

import numpy as np
import mujoco
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from camera_utils import CameraProcessor


def capture_rgbd(model, data, camera_name, width=640, height=480):
    """Capture RGB-D from specified camera."""
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Capture RGB
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    # Capture depth
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    return rgb, depth


def save_pointcloud_ply(points, colors, filepath):
    """Save point cloud in PLY format for visualization."""
    n_points = len(points)

    with open(filepath, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write data
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i].astype(int)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print(f"Saved point cloud to {filepath}")


def visualize_pointcloud(points, colors, title="Point Cloud"):
    """Visualize point cloud using matplotlib 3D scatter."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample for visualization if too many points
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points = points[indices]
        colors = colors[indices]

    # Plot
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors/255.0, s=1, marker='.')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                          points[:, 1].max()-points[:, 1].min(),
                          points[:, 2].max()-points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate point cloud from robot head camera')
    parser.add_argument('--camera', type=str, default='track_front',
                       choices=['track_front', 'track_left', 'track_right', 'right_wrist_camera', 'right_wrist_camera_down'],
                       help='Camera to use (default: track_front)')
    parser.add_argument('--width', type=int, default=640,
                       help='Image width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Image height (default: 480)')
    parser.add_argument('--output', type=str, default='./pointcloud_data',
                       help='Output directory (default: ./pointcloud_data)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize point cloud after generation')
    parser.add_argument('--min-depth', type=float, default=0.01,
                       help='Minimum depth in meters (default: 0.01)')
    parser.add_argument('--max-depth', type=float, default=2.0,
                       help='Maximum depth in meters (default: 2.0)')

    args = parser.parse_args()

    # Load scene
    scene_path = Path(__file__).parent / "hammer_grasp_rgbd_scene.xml"
    if not scene_path.exists():
        print(f"ERROR: Scene file not found: {scene_path}")
        return 1

    print(f"Loading scene from: {scene_path}")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Initialize simulation (let hammer fall and settle on table)
    print("Initializing simulation (letting hammer fall)...")

    # Get hammer body ID
    hammer_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hammer")

    # Run simulation and monitor hammer position
    for i in range(1000):
        mujoco.mj_step(model, data)

        if i % 100 == 0 and hammer_body_id >= 0:
            hammer_pos = data.xpos[hammer_body_id]
            hammer_vel = np.linalg.norm(data.cvel[hammer_body_id, :3])
            print(f"  Step {i}: Hammer at [{hammer_pos[0]:.3f}, {hammer_pos[1]:.3f}, {hammer_pos[2]:.3f}], vel={hammer_vel:.4f}")

    # Final hammer position
    if hammer_body_id >= 0:
        hammer_pos = data.xpos[hammer_body_id]
        print(f"Final hammer position: [{hammer_pos[0]:.3f}, {hammer_pos[1]:.3f}, {hammer_pos[2]:.3f}]")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Capture RGB-D
    print(f"Capturing RGB-D from camera: {args.camera}")
    rgb, depth = capture_rgbd(model, data, args.camera, args.width, args.height)

    print(f"RGB shape: {rgb.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: {depth.min():.3f} - {depth.max():.3f} meters")

    # Save RGB and depth images
    plt.imsave(output_dir / "rgb.png", rgb)
    plt.imsave(output_dir / "depth.png", depth, cmap='viridis')
    print(f"Saved RGB and depth images to {output_dir}")

    # Generate point cloud
    print("Generating point cloud...")
    processor = CameraProcessor(model, width=args.width, height=args.height)
    points, colors = processor.rgbd_to_pointcloud(
        rgb, depth, args.camera,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )

    print(f"Point cloud generated: {len(points)} points")

    if len(points) == 0:
        print("WARNING: No points in point cloud! Check depth values.")
        return 1

    # Transform to world frame
    points_world = processor.camera_to_world_frame(points, data, args.camera)

    # Save point cloud
    ply_path = output_dir / "pointcloud.ply"
    save_pointcloud_ply(points_world, colors, ply_path)

    # Save as numpy array
    npz_path = output_dir / "pointcloud.npz"
    np.savez(npz_path, points=points_world, colors=colors)
    print(f"Saved point cloud data to {npz_path}")

    # Print statistics
    print("\nPoint cloud statistics:")
    print(f"  Number of points: {len(points_world)}")
    print(f"  X range: {points_world[:, 0].min():.3f} to {points_world[:, 0].max():.3f} m")
    print(f"  Y range: {points_world[:, 1].min():.3f} to {points_world[:, 1].max():.3f} m")
    print(f"  Z range: {points_world[:, 2].min():.3f} to {points_world[:, 2].max():.3f} m")

    # Get hammer position for reference
    hammer_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hammer")
    if hammer_body_id >= 0:
        hammer_pos = data.xpos[hammer_body_id]
        print(f"\nHammer position (world frame): [{hammer_pos[0]:.3f}, {hammer_pos[1]:.3f}, {hammer_pos[2]:.3f}]")

    # Visualize if requested
    if args.visualize:
        print("\nVisualizing point cloud...")
        visualize_pointcloud(points_world, colors,
                           title=f"Point Cloud from {args.camera}")

    print("\nDone! Point cloud saved to:")
    print(f"  PLY: {ply_path}")
    print(f"  NPZ: {npz_path}")
    print(f"\nYou can view the PLY file in tools like MeshLab, CloudCompare, or Open3D")

    return 0


if __name__ == "__main__":
    sys.exit(main())
