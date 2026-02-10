#!/usr/bin/env python3
"""
Detect object and extract full geometric information.

This script demonstrates how to:
1. Detect an object using Grounding DINO
2. Extract 2D geometry (bounding box, size, position)
3. Extract 3D geometry (position, depth)
4. Estimate orientation and dimensions
"""

import numpy as np
import cv2
from pathlib import Path
from grounding_dino_detector import GroundingDINODetector


class ObjectGeometry:
    """Container for object geometric information."""

    def __init__(self, detection, depth_image=None, camera_intrinsics=None):
        """
        Args:
            detection: Detection dict from GroundingDINODetector
            depth_image: Optional depth map for 3D geometry
            camera_intrinsics: Optional 3x3 camera matrix
        """
        self.detection = detection
        self.depth_image = depth_image
        self.camera_intrinsics = camera_intrinsics

        # Extract 2D geometry
        self._extract_2d_geometry()

        # Extract 3D geometry if depth available
        if depth_image is not None and camera_intrinsics is not None:
            self._extract_3d_geometry()

    def _extract_2d_geometry(self):
        """Extract 2D geometric properties."""
        bbox = self.detection['bbox']
        x1, y1, x2, y2 = bbox

        self.bbox = bbox
        self.center_2d = self.detection['center']
        self.width_px = x2 - x1
        self.height_px = y2 - y1
        self.area_px = self.width_px * self.height_px
        self.aspect_ratio = max(self.width_px, self.height_px) / max(min(self.width_px, self.height_px), 1)

        # Estimate orientation from aspect ratio
        if self.width_px > self.height_px * 1.2:
            self.orientation_2d = "horizontal"
        elif self.height_px > self.width_px * 1.2:
            self.orientation_2d = "vertical"
        else:
            self.orientation_2d = "square"

    def _extract_3d_geometry(self):
        """Extract 3D geometric properties from depth."""
        detector = GroundingDINODetector()

        # Get 3D position
        self.position_3d = detector.get_3d_position(
            self.detection,
            self.depth_image,
            self.camera_intrinsics
        )

        # Get depth at center and corners
        cx, cy = self.center_2d
        x1, y1, x2, y2 = self.bbox

        self.depth_center = self.depth_image[cy, cx]
        self.depth_min = self.depth_image[y1:y2, x1:x2].min()
        self.depth_max = self.depth_image[y1:y2, x1:x2].max()
        self.depth_mean = self.depth_image[y1:y2, x1:x2].mean()
        self.depth_std = self.depth_image[y1:y2, x1:x2].std()

        # Estimate physical size (requires depth)
        # width_m = (width_px / fx) * depth
        # height_m = (height_px / fy) * depth
        K = self.camera_intrinsics
        fx, fy = K[0, 0], K[1, 1]

        self.width_m = (self.width_px / fx) * self.depth_center
        self.height_m = (self.height_px / fy) * self.depth_center

        # Object extent in depth
        self.depth_extent = self.depth_max - self.depth_min

    def print_summary(self):
        """Print geometric summary."""
        print(f"\n{'='*60}")
        print(f"Object Geometry: {self.detection['label']}")
        print(f"Confidence: {self.detection['confidence']:.2%}")
        print(f"{'='*60}")

        print(f"\n2D Geometry (Image Space):")
        print(f"  Bounding Box: {self.bbox}")
        print(f"  Center: {self.center_2d} px")
        print(f"  Size: {self.width_px} x {self.height_px} px")
        print(f"  Area: {self.area_px} px²")
        print(f"  Aspect Ratio: {self.aspect_ratio:.2f}")
        print(f"  Orientation: {self.orientation_2d}")

        if hasattr(self, 'position_3d'):
            print(f"\n3D Geometry (World Space):")
            print(f"  Position: [{self.position_3d[0]:.3f}, {self.position_3d[1]:.3f}, {self.position_3d[2]:.3f}] m")
            print(f"  Depth (center): {self.depth_center:.3f} m")
            print(f"  Depth (range): {self.depth_min:.3f} - {self.depth_max:.3f} m")
            print(f"  Depth (std): {self.depth_std:.4f} m")

            print(f"\nEstimated Physical Size:")
            print(f"  Width: {self.width_m*100:.1f} cm")
            print(f"  Height: {self.height_m*100:.1f} cm")
            print(f"  Depth extent: {self.depth_extent*100:.1f} cm")

    def to_dict(self):
        """Convert to dictionary for export."""
        result = {
            'label': self.detection['label'],
            'confidence': float(self.detection['confidence']),
            '2d': {
                'bbox': self.bbox,
                'center': self.center_2d,
                'width_px': int(self.width_px),
                'height_px': int(self.height_px),
                'area_px': int(self.area_px),
                'aspect_ratio': float(self.aspect_ratio),
                'orientation': self.orientation_2d
            }
        }

        if hasattr(self, 'position_3d'):
            result['3d'] = {
                'position': self.position_3d.tolist(),
                'depth_center': float(self.depth_center),
                'depth_range': [float(self.depth_min), float(self.depth_max)],
                'depth_std': float(self.depth_std),
                'width_m': float(self.width_m),
                'height_m': float(self.height_m),
                'depth_extent_m': float(self.depth_extent)
            }

        return result


def detect_and_analyze_geometry(
    rgb_image_path,
    text_prompt="hammer",
    depth_image_path=None,
    camera_intrinsics=None
):
    """
    Detect object and analyze its geometry.

    Args:
        rgb_image_path: Path to RGB image
        text_prompt: What to detect
        depth_image_path: Optional path to depth image
        camera_intrinsics: Optional 3x3 camera matrix

    Returns:
        List of ObjectGeometry instances
    """
    # Initialize detector
    print("Initializing Grounding DINO...")
    detector = GroundingDINODetector()

    # Load RGB
    print(f"Loading image: {rgb_image_path}")
    rgb = cv2.imread(str(rgb_image_path))
    if rgb is None:
        raise ValueError(f"Failed to load image: {rgb_image_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Load depth if provided
    depth = None
    if depth_image_path:
        print(f"Loading depth: {depth_image_path}")
        if str(depth_image_path).endswith('.npy'):
            depth = np.load(depth_image_path)
        else:
            depth_img = cv2.imread(str(depth_image_path), cv2.IMREAD_UNCHANGED)
            # Convert to meters (adjust based on your depth format)
            depth = depth_img.astype(np.float32) / 1000.0

    # Detect objects
    print(f"\nDetecting '{text_prompt}'...")
    detections = detector.detect(rgb, text_prompt=text_prompt, box_threshold=0.25)

    print(f"Found {len(detections)} detection(s)")

    # Analyze geometry for each detection
    geometries = []
    for i, det in enumerate(detections):
        print(f"\n--- Detection {i+1}/{len(detections)} ---")
        geom = ObjectGeometry(det, depth, camera_intrinsics)
        geom.print_summary()
        geometries.append(geom)

    # Visualize
    if len(detections) > 0:
        output_path = "geometry_detection_result.png"
        detector.visualize_detections(rgb, detections, save_path=output_path)
        print(f"\n✓ Saved visualization to {output_path}")

    return geometries


if __name__ == "__main__":
    # Example 1: 2D geometry only (no depth)
    print("="*60)
    print("Example: Detect Object Geometry")
    print("="*60)

    geometries = detect_and_analyze_geometry(
        rgb_image_path="pointcloud_data/rgb.png",
        text_prompt="hammer"
    )

    # Example 2: Full 3D geometry (with depth)
    # Define camera intrinsics (adjust for your camera!)
    # K = np.array([
    #     [525.0, 0, 320.0],
    #     [0, 525.0, 240.0],
    #     [0, 0, 1.0]
    # ])
    #
    # geometries = detect_and_analyze_geometry(
    #     rgb_image_path="pointcloud_data/rgb.png",
    #     text_prompt="hammer",
    #     depth_image_path="pointcloud_data/depth.png",
    #     camera_intrinsics=K
    # )

    # Export to JSON
    if len(geometries) > 0:
        import json
        output_data = [g.to_dict() for g in geometries]
        with open("object_geometry.json", "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Saved geometry data to object_geometry.json")
