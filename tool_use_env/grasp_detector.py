#!/usr/bin/env python3
"""
Grasp detection module with support for multiple backends.

Supports:
- AnyGrasp (if installed)
- GraspNet (if installed)
- Heuristic fallback (geometry-based)
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


class Grasp:
    """Represents a 6-DOF grasp pose."""

    def __init__(self,
                 position: np.ndarray,
                 orientation: np.ndarray,
                 width: float = 0.08,
                 score: float = 1.0,
                 object_id: Optional[int] = None):
        """
        Initialize grasp.

        Args:
            position: (3,) grasp position [x, y, z]
            orientation: (3, 3) rotation matrix or (4,) quaternion [w,x,y,z]
            width: Gripper opening width in meters
            score: Confidence score [0, 1]
            object_id: Optional object ID
        """
        self.position = np.array(position, dtype=np.float32)

        # Handle both rotation matrix and quaternion
        if orientation.shape == (3, 3):
            self.rotation_matrix = orientation.astype(np.float32)
            self.quaternion = self._matrix_to_quat(orientation)
        elif orientation.shape == (4,):
            self.quaternion = orientation.astype(np.float32)
            self.rotation_matrix = self._quat_to_matrix(orientation)
        else:
            raise ValueError(f"Invalid orientation shape: {orientation.shape}")

        self.width = float(width)
        self.score = float(score)
        self.object_id = object_id

    @staticmethod
    def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w,x,y,z] to rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ], dtype=np.float32)

    @staticmethod
    def _matrix_to_quat(mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w,x,y,z]."""
        trace = np.trace(mat)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (mat[2,1] - mat[1,2]) * s
            y = (mat[0,2] - mat[2,0]) * s
            z = (mat[1,0] - mat[0,1]) * s
        else:
            if mat[0,0] > mat[1,1] and mat[0,0] > mat[2,2]:
                s = 2.0 * np.sqrt(1.0 + mat[0,0] - mat[1,1] - mat[2,2])
                w = (mat[2,1] - mat[1,2]) / s
                x = 0.25 * s
                y = (mat[0,1] + mat[1,0]) / s
                z = (mat[0,2] + mat[2,0]) / s
            elif mat[1,1] > mat[2,2]:
                s = 2.0 * np.sqrt(1.0 + mat[1,1] - mat[0,0] - mat[2,2])
                w = (mat[0,2] - mat[2,0]) / s
                x = (mat[0,1] + mat[1,0]) / s
                y = 0.25 * s
                z = (mat[1,2] + mat[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + mat[2,2] - mat[0,0] - mat[1,1])
                w = (mat[1,0] - mat[0,1]) / s
                x = (mat[0,2] + mat[2,0]) / s
                y = (mat[1,2] + mat[2,1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z], dtype=np.float32)

    def to_pose_array(self) -> np.ndarray:
        """Return grasp as 7D array [x, y, z, qw, qx, qy, qz]."""
        return np.concatenate([self.position, self.quaternion])

    def __repr__(self):
        return (f"Grasp(pos={self.position}, "
                f"width={self.width:.3f}, score={self.score:.3f})")


class GraspDetector:
    """Base class for grasp detection."""

    def detect(self,
              points: np.ndarray,
              colors: Optional[np.ndarray] = None,
              **kwargs) -> List[Grasp]:
        """
        Detect grasps from point cloud.

        Args:
            points: (N, 3) 3D points
            colors: Optional (N, 3) RGB colors
            **kwargs: Backend-specific arguments

        Returns:
            List of Grasp objects sorted by score (descending)
        """
        raise NotImplementedError


class HeuristicGraspDetector(GraspDetector):
    """
    Simple geometry-based grasp detector.

    Uses surface normals and clustering to find graspable regions.
    Useful as fallback when VLM models are not available.
    """

    def __init__(self, gripper_width: float = 0.08):
        """
        Initialize heuristic detector.

        Args:
            gripper_width: Maximum gripper opening (meters)
        """
        self.gripper_width = gripper_width

    def detect(self,
              points: np.ndarray,
              colors: Optional[np.ndarray] = None,
              num_grasps: int = 10,
              **kwargs) -> List[Grasp]:
        """
        Detect grasps using geometric heuristics.

        Args:
            points: (N, 3) point cloud
            colors: Optional (N, 3) colors
            num_grasps: Number of grasp candidates to return

        Returns:
            List of Grasp objects
        """
        if len(points) < 10:
            warnings.warn("Too few points for grasp detection")
            return []

        # Compute surface normals using PCA on local neighborhoods
        normals = self._estimate_normals(points)

        # Find highest points (potential grasps from above)
        z_threshold = np.percentile(points[:, 2], 80)
        high_points = points[points[:, 2] > z_threshold]

        if len(high_points) == 0:
            high_points = points

        # Sample grasp candidates
        grasps = []
        num_samples = min(num_grasps * 3, len(high_points))
        indices = np.random.choice(len(high_points), num_samples, replace=False)

        for idx in indices:
            pos = high_points[idx]

            # Grasp orientation: approach from above, gripper horizontal
            # Z-axis points down (approach direction)
            # X-axis is grasp closing direction
            approach = np.array([0, 0, -1])  # Down
            closing = np.array([1, 0, 0])    # Along X
            lateral = np.cross(approach, closing)

            rotation = np.column_stack([closing, lateral, approach])

            # Score based on height (prefer higher grasps)
            height_score = (pos[2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())

            # Score based on point density (prefer dense regions)
            distances = np.linalg.norm(points - pos, axis=1)
            density_score = np.sum(distances < 0.05) / len(points)

            score = 0.7 * height_score + 0.3 * density_score

            grasp = Grasp(
                position=pos,
                orientation=rotation,
                width=self.gripper_width,
                score=score
            )
            grasps.append(grasp)

        # Sort by score and return top candidates
        grasps.sort(key=lambda g: g.score, reverse=True)
        return grasps[:num_grasps]

    def _estimate_normals(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """Estimate surface normals using local PCA."""
        normals = np.zeros_like(points)

        for i in range(len(points)):
            # Find k nearest neighbors
            distances = np.linalg.norm(points - points[i], axis=1)
            neighbors = points[np.argsort(distances)[:k]]

            # PCA on neighbors
            centered = neighbors - neighbors.mean(axis=0)
            _, _, vh = np.linalg.svd(centered)

            # Normal is the direction of minimum variance
            normal = vh[2]

            # Orient normals consistently (point upward)
            if normal[2] < 0:
                normal = -normal

            normals[i] = normal

        return normals


class AnyGraspDetector(GraspDetector):
    """
    AnyGrasp-based grasp detector.

    Requires anygrasp SDK to be installed.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize AnyGrasp detector.

        Args:
            checkpoint_path: Path to AnyGrasp checkpoint
        """
        try:
            # Try to import anygrasp
            from anygrasp import AnyGrasp
            self.detector = AnyGrasp(checkpoint=checkpoint_path)
            self.available = True
            print("✓ AnyGrasp detector loaded")
        except ImportError:
            warnings.warn("AnyGrasp not installed. Install with: pip install anygrasp-sdk")
            self.available = False
            self.detector = None

    def detect(self,
              points: np.ndarray,
              colors: Optional[np.ndarray] = None,
              num_grasps: int = 10,
              **kwargs) -> List[Grasp]:
        """
        Detect grasps using AnyGrasp.

        Args:
            points: (N, 3) point cloud
            colors: Optional (N, 3) RGB colors
            num_grasps: Number of grasps to return

        Returns:
            List of Grasp objects
        """
        if not self.available:
            raise RuntimeError("AnyGrasp not available")

        # Run detection
        grasp_results = self.detector.predict(points, colors, num_grasp=num_grasps)

        # Convert to Grasp objects
        grasps = []
        for i in range(len(grasp_results)):
            grasp = Grasp(
                position=grasp_results[i]['translation'],
                orientation=grasp_results[i]['rotation_matrix'],
                width=grasp_results[i]['width'],
                score=grasp_results[i]['score']
            )
            grasps.append(grasp)

        return grasps


class GraspNetDetector(GraspDetector):
    """
    GraspNet-based grasp detector.

    Requires graspnetAPI to be installed.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize GraspNet detector.

        Args:
            checkpoint_path: Path to GraspNet checkpoint
        """
        try:
            from graspnetAPI import GraspNet
            self.detector = GraspNet(checkpoint=checkpoint_path)
            self.available = True
            print("✓ GraspNet detector loaded")
        except ImportError:
            warnings.warn("GraspNet not installed. Install with: pip install graspnetAPI")
            self.available = False
            self.detector = None

    def detect(self,
              points: np.ndarray,
              colors: Optional[np.ndarray] = None,
              num_grasps: int = 10,
              **kwargs) -> List[Grasp]:
        """
        Detect grasps using GraspNet.

        Args:
            points: (N, 3) point cloud
            colors: Optional (N, 3) RGB colors
            num_grasps: Number of grasps to return

        Returns:
            List of Grasp objects
        """
        if not self.available:
            raise RuntimeError("GraspNet not available")

        # Run detection
        grasp_results = self.detector.detect(points, colors, top_k=num_grasps)

        # Convert to Grasp objects
        grasps = []
        for result in grasp_results:
            grasp = Grasp(
                position=result['center'],
                orientation=result['rotation'],
                width=result['width'],
                score=result['score']
            )
            grasps.append(grasp)

        return grasps


def create_grasp_detector(backend: str = 'auto',
                         checkpoint_path: Optional[str] = None) -> GraspDetector:
    """
    Factory function to create grasp detector.

    Args:
        backend: One of 'auto', 'anygrasp', 'graspnet', 'heuristic'
        checkpoint_path: Optional path to model checkpoint

    Returns:
        GraspDetector instance
    """
    if backend == 'auto':
        # Try AnyGrasp first, then GraspNet, then heuristic
        try:
            detector = AnyGraspDetector(checkpoint_path)
            if detector.available:
                return detector
        except:
            pass

        try:
            detector = GraspNetDetector(checkpoint_path)
            if detector.available:
                return detector
        except:
            pass

        print("⚠ No VLM grasp detector available, using heuristic fallback")
        return HeuristicGraspDetector()

    elif backend == 'anygrasp':
        return AnyGraspDetector(checkpoint_path)
    elif backend == 'graspnet':
        return GraspNetDetector(checkpoint_path)
    elif backend == 'heuristic':
        return HeuristicGraspDetector()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Example usage
if __name__ == "__main__":
    print("Grasp detector module loaded successfully")
    print("\nAvailable backends:")
    print("  - 'anygrasp': AnyGrasp VLM (requires anygrasp-sdk)")
    print("  - 'graspnet': GraspNet (requires graspnetAPI)")
    print("  - 'heuristic': Geometry-based fallback (always available)")
    print("  - 'auto': Automatically select best available")

    # Test heuristic detector
    print("\nTesting heuristic detector...")
    detector = create_grasp_detector('heuristic')

    # Generate dummy point cloud
    dummy_points = np.random.randn(1000, 3) * 0.1
    dummy_points[:, 2] += 0.5  # Shift up

    grasps = detector.detect(dummy_points, num_grasps=5)
    print(f"✓ Generated {len(grasps)} grasp candidates")
    for i, grasp in enumerate(grasps[:3]):
        print(f"  Grasp {i+1}: {grasp}")
