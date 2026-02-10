#!/usr/bin/env python3
"""
Visualize the detected grasp on the RGB image.

Shows the best grasp pose overlaid on the camera view.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def project_point_to_image(point_3d, K, camera_pose):
    """Project 3D point to 2D image coordinates."""
    # Transform from world to camera frame
    # (simplified - assumes camera_pose is transformation matrix)
    point_cam = point_3d  # Already in camera frame if using camera coordinates

    # Project using intrinsic matrix
    point_2d_h = K @ point_cam
    u = int(point_2d_h[0] / point_2d_h[2])
    v = int(point_2d_h[1] / point_2d_h[2])

    return u, v


def visualize_grasp_on_image():
    """Create visualization of grasp on RGB image."""
    data_dir = Path("./pointcloud_data")

    # Load RGB image
    rgb_path = data_dir / "rgb.png"
    rgb = plt.imread(rgb_path)

    # Load grasp data
    grasp_path = data_dir / "detected_grasps.npz"
    grasp_data = np.load(grasp_path)

    positions = grasp_data['positions']
    scores = grasp_data['scores']

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Original RGB
    axes[0].imshow(rgb)
    axes[0].set_title('Robot\'s View (track_front camera)')
    axes[0].axis('off')

    # Plot 2: RGB with grasp annotation
    axes[1].imshow(rgb)

    # Annotate best grasp position
    best_grasp_pos = positions[0]
    best_score = scores[0]

    # Add text annotation (approximate position)
    axes[1].text(320, 240, f"Best Grasp Detected",
                fontsize=14, color='lime', weight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

    axes[1].text(320, 270,
                f"Position: [{best_grasp_pos[0]:.3f}, {best_grasp_pos[1]:.3f}, {best_grasp_pos[2]:.3f}]",
                fontsize=10, color='white', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    axes[1].text(320, 290,
                f"Confidence: {best_score:.1%}",
                fontsize=10, color='yellow', ha='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # Draw crosshair at center
    axes[1].plot([320], [280], 'g+', markersize=30, markeredgewidth=3)

    axes[1].set_title('Grasp Detection Result')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(data_dir / "grasp_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {data_dir / 'grasp_visualization.png'}")
    plt.show()


if __name__ == "__main__":
    visualize_grasp_on_image()
