#!/usr/bin/env python3
"""
Segment the hammer from the point cloud based on color and geometry.

This script identifies the hammer in the point cloud by filtering for
its characteristic brown/orange color and spatial location.

Usage:
    python segment_hammer.py --pointcloud ./pointcloud_data/pointcloud.npz --visualize
"""

import numpy as np
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def segment_by_color(points, colors, target_color_ranges):
    """
    Segment points based on color ranges.

    Args:
        points: (N, 3) point cloud
        colors: (N, 3) RGB colors (0-255)
        target_color_ranges: List of (min_rgb, max_rgb) tuples

    Returns:
        mask: Boolean mask for points matching color criteria
    """
    mask = np.zeros(len(points), dtype=bool)

    for min_rgb, max_rgb in target_color_ranges:
        in_range = np.all((colors >= min_rgb) & (colors <= max_rgb), axis=1)
        mask |= in_range

    return mask


def segment_by_height(points, min_height, max_height):
    """Segment points within height range."""
    return (points[:, 2] >= min_height) & (points[:, 2] <= max_height)


def filter_outliers(points, colors, k=50, std_ratio=1.0):
    """Remove statistical outliers from point cloud."""
    from scipy.spatial import KDTree

    if len(points) < k:
        return points, colors

    # Build KD-tree
    tree = KDTree(points)

    # Compute mean distance to k nearest neighbors
    distances, _ = tree.query(points, k=k+1)  # +1 because point itself is included
    mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self

    # Filter based on statistical threshold
    threshold = np.mean(mean_distances) + std_ratio * np.std(mean_distances)
    mask = mean_distances < threshold

    return points[mask], colors[mask]


def compute_bounding_box(points):
    """Compute axis-aligned bounding box."""
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    return min_bound, max_bound, center, size


def visualize_segmentation(all_points, all_colors, hammer_points, hammer_colors, title="Hammer Segmentation"):
    """Visualize original and segmented point clouds."""
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Original point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    if len(all_points) > 10000:
        indices = np.random.choice(len(all_points), 10000, replace=False)
        viz_points = all_points[indices]
        viz_colors = all_colors[indices]
    else:
        viz_points = all_points
        viz_colors = all_colors

    ax1.scatter(viz_points[:, 0], viz_points[:, 1], viz_points[:, 2],
               c=viz_colors/255.0, s=1, marker='.')
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')

    # Plot 2: Segmented hammer
    ax2 = fig.add_subplot(132, projection='3d')
    if len(hammer_points) > 0:
        ax2.scatter(hammer_points[:, 0], hammer_points[:, 1], hammer_points[:, 2],
                   c=hammer_colors/255.0, s=2, marker='.')
        ax2.set_title(f'Segmented Hammer ({len(hammer_points)} points)')
    else:
        ax2.set_title('No hammer points found')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')

    # Plot 3: Highlighted in context
    ax3 = fig.add_subplot(133, projection='3d')
    # Show sparse original in gray
    if len(all_points) > 5000:
        indices = np.random.choice(len(all_points), 5000, replace=False)
        ax3.scatter(all_points[indices, 0], all_points[indices, 1], all_points[indices, 2],
                   c='gray', s=0.5, alpha=0.1, marker='.')

    # Highlight hammer in red
    if len(hammer_points) > 0:
        ax3.scatter(hammer_points[:, 0], hammer_points[:, 1], hammer_points[:, 2],
                   c='red', s=3, marker='.')
    ax3.set_title('Hammer Highlighted')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Segment hammer from point cloud')
    parser.add_argument('--pointcloud', type=str, default='./pointcloud_data/pointcloud.npz',
                       help='Path to point cloud NPZ file')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize segmentation result')
    parser.add_argument('--min-height', type=float, default=2.0,
                       help='Minimum height for hammer (default: 2.0m)')
    parser.add_argument('--max-height', type=float, default=2.4,
                       help='Maximum height for hammer (default: 2.4m)')

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

    # Define hammer color ranges (brown/orange/tan)
    # Hammer in the scene appears brown/orange
    hammer_color_ranges = [
        (np.array([100, 60, 30]), np.array([220, 160, 120])),   # Brown range
        (np.array([150, 80, 40]), np.array([255, 180, 140])),   # Orange-brown
        (np.array([120, 90, 60]), np.array([200, 170, 130])),   # Tan
    ]

    print("\nSegmenting hammer by color...")
    color_mask = segment_by_color(points, colors, hammer_color_ranges)
    print(f"  Found {color_mask.sum()} points matching hammer color")

    print(f"\nSegmenting by height ({args.min_height}m - {args.max_height}m)...")
    height_mask = segment_by_height(points, args.min_height, args.max_height)
    print(f"  Found {height_mask.sum()} points in height range")

    # Combine masks
    hammer_mask = color_mask & height_mask
    hammer_points = points[hammer_mask]
    hammer_colors = colors[hammer_mask]

    print(f"\nCombined segmentation: {len(hammer_points)} points")

    if len(hammer_points) == 0:
        print("\nWARNING: No hammer points found!")
        print("Possible issues:")
        print("  - Hammer color not matching expected ranges")
        print("  - Height range too restrictive")
        print("  - Hammer not visible in point cloud")
        print("\nShowing color distribution in point cloud:")

        # Analyze color distribution
        print(f"  R range: [{colors[:, 0].min()}, {colors[:, 0].max()}]")
        print(f"  G range: [{colors[:, 1].min()}, {colors[:, 1].max()}]")
        print(f"  B range: [{colors[:, 2].min()}, {colors[:, 2].max()}]")

        # Show height distribution
        print(f"\nHeight (Z) distribution:")
        for percentile in [0, 25, 50, 75, 100]:
            val = np.percentile(points[:, 2], percentile)
            print(f"  {percentile:3d}%: {val:.3f}m")

        return 1

    # Compute bounding box
    min_bound, max_bound, center, size = compute_bounding_box(hammer_points)
    print(f"\nHammer bounding box:")
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
    print(f"  Min: [{min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}]")
    print(f"  Max: [{max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f}]")

    # Save segmented hammer
    output_path = pointcloud_path.parent / "hammer_segmented.npz"
    np.savez(output_path,
             points=hammer_points,
             colors=hammer_colors,
             center=center,
             size=size,
             min_bound=min_bound,
             max_bound=max_bound)
    print(f"\nSaved segmented hammer to: {output_path}")

    # Visualize if requested
    if args.visualize:
        print("\nVisualizing segmentation...")
        visualize_segmentation(points, colors, hammer_points, hammer_colors)

    # Print recommendations
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Use the segmented hammer for grasp detection:")
    print(f"   python analyze_grasps.py --pointcloud {output_path}")
    print("\n2. The hammer center position is:")
    print(f"   [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print("\n3. Recommended grasp approach:")
    print("   - Approach from above or side")
    print("   - Grasp handle (upper part of hammer)")
    print("   - Gripper width should be ~0.08m")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
