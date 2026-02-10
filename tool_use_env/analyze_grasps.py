#!/usr/bin/env python3
"""
Analyze point cloud and detect grasp poses for the hammer.

This script loads the generated point cloud and uses grasp detection
to find suitable grasp poses for the robot.

Usage:
    python analyze_grasps.py --pointcloud ./pointcloud_data/pointcloud.npz --visualize
"""

import numpy as np
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from grasp_detector import create_grasp_detector


def visualize_grasps(points, colors, grasps, title="Grasp Candidates"):
    """Visualize point cloud with detected grasps."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample points for visualization
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points_viz = points[indices]
        colors_viz = colors[indices]
    else:
        points_viz = points
        colors_viz = colors

    # Plot point cloud
    ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2],
               c=colors_viz/255.0, s=1, marker='.')

    # Plot grasps as coordinate frames
    arrow_length = 0.05  # 5cm arrows

    for i, grasp in enumerate(grasps[:10]):  # Show top 10 grasps
        pos = grasp.position
        rot = grasp.rotation_matrix

        # X-axis (red) - approach direction
        ax.quiver(pos[0], pos[1], pos[2],
                 rot[0, 0]*arrow_length, rot[1, 0]*arrow_length, rot[2, 0]*arrow_length,
                 color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)

        # Y-axis (green) - gripper opening direction
        ax.quiver(pos[0], pos[1], pos[2],
                 rot[0, 1]*arrow_length, rot[1, 1]*arrow_length, rot[2, 1]*arrow_length,
                 color='green', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)

        # Z-axis (blue) - gripper closing direction
        ax.quiver(pos[0], pos[1], pos[2],
                 rot[0, 2]*arrow_length, rot[1, 2]*arrow_length, rot[2, 2]*arrow_length,
                 color='blue', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)

        # Label with score
        ax.text(pos[0], pos[1], pos[2] + 0.03,
               f"#{i+1} ({grasp.score:.2f})",
               fontsize=8, color='black', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

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

    plt.tight_layout()
    plt.show()


def filter_points_around_hammer(points, colors, hammer_pos, radius=0.3):
    """Filter point cloud to only include points near the hammer."""
    # Calculate distances to hammer
    distances = np.linalg.norm(points - hammer_pos, axis=1)

    # Keep only points within radius
    mask = distances < radius
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    print(f"Filtered {len(filtered_points)} points around hammer (within {radius}m radius)")

    return filtered_points, filtered_colors


def main():
    parser = argparse.ArgumentParser(description='Analyze point cloud and detect grasps')
    parser.add_argument('--pointcloud', type=str, default='./pointcloud_data/pointcloud.npz',
                       help='Path to point cloud NPZ file')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize detected grasps')
    parser.add_argument('--num-grasps', type=int, default=10,
                       help='Number of grasp candidates to detect (default: 10)')
    parser.add_argument('--backend', type=str, default='heuristic',
                       choices=['heuristic', 'anygrasp', 'graspnet', 'auto'],
                       help='Grasp detection backend (default: heuristic)')
    parser.add_argument('--hammer-pos', type=float, nargs=3,
                       default=[0.9, 0.0, 2.2],
                       help='Hammer position (x y z) for filtering (default: 0.9 0.0 2.2)')
    parser.add_argument('--filter-radius', type=float, default=0.3,
                       help='Radius around hammer to filter points (default: 0.3m)')

    args = parser.parse_args()

    # Load point cloud
    pointcloud_path = Path(args.pointcloud)
    if not pointcloud_path.exists():
        print(f"ERROR: Point cloud file not found: {pointcloud_path}")
        return 1

    print(f"Loading point cloud from: {pointcloud_path}")
    data = np.load(pointcloud_path)
    points = data['points']
    colors = data['colors']

    print(f"Loaded {len(points)} points")
    print(f"Point cloud bounds:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    # Filter points around hammer
    hammer_pos = np.array(args.hammer_pos)
    print(f"\nFiltering points around hammer at [{hammer_pos[0]:.3f}, {hammer_pos[1]:.3f}, {hammer_pos[2]:.3f}]")
    filtered_points, filtered_colors = filter_points_around_hammer(
        points, colors, hammer_pos, args.filter_radius
    )

    if len(filtered_points) < 100:
        print("WARNING: Very few points near hammer. Consider:")
        print("  - Increasing --filter-radius")
        print("  - Adjusting --hammer-pos")
        print("  - Checking if hammer is visible in point cloud")

    # Create grasp detector
    print(f"\nCreating grasp detector (backend: {args.backend})...")
    detector = create_grasp_detector(args.backend)

    # Detect grasps
    print(f"Detecting {args.num_grasps} grasp candidates...")
    grasps = detector.detect(filtered_points, filtered_colors, num_grasps=args.num_grasps)

    print(f"\nDetected {len(grasps)} grasps:")
    print("-" * 60)
    for i, grasp in enumerate(grasps):
        print(f"\nGrasp #{i+1}:")
        print(f"  Position: [{grasp.position[0]:.3f}, {grasp.position[1]:.3f}, {grasp.position[2]:.3f}]")
        print(f"  Quaternion (wxyz): [{grasp.quaternion[0]:.3f}, {grasp.quaternion[1]:.3f}, "
              f"{grasp.quaternion[2]:.3f}, {grasp.quaternion[3]:.3f}]")
        print(f"  Width: {grasp.width:.3f} m")
        print(f"  Score: {grasp.score:.3f}")

    # Save grasp data
    output_path = pointcloud_path.parent / "detected_grasps.npz"
    grasp_data = {
        'positions': np.array([g.position for g in grasps]),
        'quaternions': np.array([g.quaternion for g in grasps]),
        'widths': np.array([g.width for g in grasps]),
        'scores': np.array([g.score for g in grasps])
    }
    np.savez(output_path, **grasp_data)
    print(f"\nSaved grasp data to: {output_path}")

    # Visualize if requested
    if args.visualize:
        print("\nVisualizing grasps...")
        visualize_grasps(filtered_points, filtered_colors, grasps,
                        title=f"Top {len(grasps)} Grasp Candidates")

    # Print best grasp recommendation
    if len(grasps) > 0:
        best_grasp = grasps[0]
        print("\n" + "="*60)
        print("RECOMMENDED GRASP (Best Score):")
        print("="*60)
        print(f"Position: [{best_grasp.position[0]:.4f}, {best_grasp.position[1]:.4f}, {best_grasp.position[2]:.4f}]")
        print(f"Quaternion (wxyz): [{best_grasp.quaternion[0]:.4f}, {best_grasp.quaternion[1]:.4f}, "
              f"{best_grasp.quaternion[2]:.4f}, {best_grasp.quaternion[3]:.4f}]")
        print(f"Gripper width: {best_grasp.width:.4f} m")
        print(f"Confidence score: {best_grasp.score:.4f}")
        print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
