#!/usr/bin/env python3
"""
Segment hammer by spatial clustering and position in camera view.
"""

import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def segment_by_color(points, colors, target_color_ranges):
    """Segment points based on color ranges."""
    mask = np.zeros(len(points), dtype=bool)
    for min_rgb, max_rgb in target_color_ranges:
        in_range = np.all((colors >= min_rgb) & (colors <= max_rgb), axis=1)
        mask |= in_range
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pointcloud', type=str, default='./pointcloud_data/pointcloud.npz')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    # Load point cloud
    data = np.load(args.pointcloud)
    points = data['points']
    colors = data['colors']

    print(f"Loaded {len(points)} points")
    print(f"Point cloud bounds:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    # Hammer is brown/orange - define color ranges
    hammer_color_ranges = [
        (np.array([80, 50, 20]), np.array([255, 200, 150])),   # Brown/orange/tan
    ]

    print("\nSegmenting by hammer color (brown/orange)...")
    color_mask = segment_by_color(points, colors, hammer_color_ranges)
    print(f"  Found {color_mask.sum()} color-matched points")

    # The hammer should be on the table, which is roughly in the middle depth
    # Filter by being in the center-ish of the scene
    color_points = points[color_mask]

    if len(color_points) == 0:
        print("No points matched hammer color!")
        return 1

    # Cluster by spatial proximity to find the largest cluster (the hammer)
    from scipy.spatial import distance_matrix
    from sklearn.cluster import DBSCAN

    print("\nClustering to isolate hammer...")
    clustering = DBSCAN(eps=0.05, min_samples=10).fit(color_points)
    labels = clustering.labels_

    # Find largest cluster
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise

    if len(unique_labels) == 0:
        print("No clusters found!")
        return 1

    cluster_sizes = {label: (labels == label).sum() for label in unique_labels}
    largest_cluster = max(cluster_sizes, key=cluster_sizes.get)

    print(f"  Found {len(unique_labels)} clusters")
    print(f"  Largest cluster: {cluster_sizes[largest_cluster]} points")

    # Extract hammer points
    cluster_mask_local = labels == largest_cluster
    hammer_points_local = color_points[cluster_mask_local]

    # Map back to original indices
    color_indices = np.where(color_mask)[0]
    hammer_indices = color_indices[cluster_mask_local]
    hammer_points = points[hammer_indices]
    hammer_colors = colors[hammer_indices]

    print(f"\nSegmented hammer: {len(hammer_points)} points")

    # Compute bounding box
    min_bound = np.min(hammer_points, axis=0)
    max_bound = np.max(hammer_points, axis=0)
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound

    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")

    # Save
    output_path = Path(args.pointcloud).parent / "hammer_segmented.npz"
    np.savez(output_path,
             points=hammer_points,
             colors=hammer_colors,
             center=center,
             size=size,
             min_bound=min_bound,
             max_bound=max_bound)
    print(f"\nSaved to {output_path}")

    # Visualize
    if args.visualize:
        fig = plt.figure(figsize=(15, 5))

        # Original with colors
        ax1 = fig.add_subplot(131, projection='3d')
        sample = np.random.choice(len(points), min(10000, len(points)), replace=False)
        ax1.scatter(points[sample, 0], points[sample, 1], points[sample, 2],
                   c=colors[sample]/255, s=1)
        ax1.set_title('Original')

        # Hammer highlighted
        ax2 = fig.add_subplot(132, projection='3d')
        sample = np.random.choice(len(points), min(5000, len(points)), replace=False)
        ax2.scatter(points[sample, 0], points[sample, 1], points[sample, 2],
                   c='gray', s=0.5, alpha=0.1)
        ax2.scatter(hammer_points[:, 0], hammer_points[:, 1], hammer_points[:, 2],
                   c='red', s=3)
        ax2.set_title('Hammer Segmented')

        # Hammer only
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(hammer_points[:, 0], hammer_points[:, 1], hammer_points[:, 2],
                   c=hammer_colors/255, s=5)
        ax3.set_title('Hammer Only')

        plt.tight_layout()
        plt.show()

    print("\n" + "="*60)
    print("Use for grasp detection:")
    print(f"  python analyze_grasps.py --pointcloud {output_path} --num-grasps 10")
    print("="*60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
