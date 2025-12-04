#!/usr/bin/env python3
"""
Camera utilities for VLM-based grasping.

Provides:
- Camera intrinsics computation
- RGB-D to point cloud conversion
- Coordinate frame transformations
- Grasp pose transformations
"""

import numpy as np
import mujoco
from typing import Tuple, Optional


class CameraProcessor:
    """Handles camera operations for grasp detection."""

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480):
        """
        Initialize camera processor.

        Args:
            model: MuJoCo model
            width: Image width in pixels
            height: Image height in pixels
        """
        self.model = model
        self.width = width
        self.height = height

    def get_camera_intrinsics(self, camera_name: str) -> np.ndarray:
        """
        Get camera intrinsic matrix.

        Args:
            camera_name: Name of camera in MuJoCo model

        Returns:
            3x3 intrinsic matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")

        # Get field of view (in degrees)
        fovy = self.model.cam_fovy[cam_id]

        # Convert to focal length (in pixels)
        # f = height / (2 * tan(fovy/2))
        f = self.height / (2.0 * np.tan(np.radians(fovy) / 2.0))

        # Principal point (assume centered)
        cx = self.width / 2.0
        cy = self.height / 2.0

        # Intrinsic matrix
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def rgbd_to_pointcloud(self,
                          rgb: np.ndarray,
                          depth: np.ndarray,
                          camera_name: str,
                          min_depth: float = 0.01,
                          max_depth: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert RGB-D image to colored point cloud.

        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) float32 in meters
            camera_name: Camera name for intrinsics
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth

        Returns:
            Tuple of (points, colors)
            - points: (N, 3) array of 3D points in camera frame
            - colors: (N, 3) array of RGB colors (0-255)
        """
        height, width = depth.shape
        K = self.get_camera_intrinsics(camera_name)

        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Back-project to 3D using pinhole camera model
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        z = depth
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]

        # Stack into point cloud (H, W, 3)
        points_3d = np.stack([x, y, z], axis=-1)

        # Flatten
        points = points_3d.reshape(-1, 3)
        colors = rgb.reshape(-1, 3)

        # Filter by depth range
        valid_mask = (depth.flatten() > min_depth) & (depth.flatten() < max_depth)
        points = points[valid_mask]
        colors = colors[valid_mask]

        return points, colors

    def get_camera_pose(self, data: mujoco.MjData, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera position and orientation in world frame.

        Args:
            data: MuJoCo data
            camera_name: Camera name

        Returns:
            Tuple of (position, rotation_matrix)
            - position: (3,) camera position in world frame
            - rotation_matrix: (3, 3) camera orientation in world frame
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        # Get camera pose from MuJoCo
        # Note: MuJoCo cameras have different conventions
        cam_pos = np.zeros(3)
        cam_mat = np.zeros(9)
        mujoco.mj_forward(self.model, data)

        # Camera position and orientation
        cam_xpos = data.cam_xpos[cam_id]
        cam_xmat = data.cam_xmat[cam_id].reshape(3, 3)

        return cam_xpos.copy(), cam_xmat.copy()

    def camera_to_world_frame(self,
                             points: np.ndarray,
                             data: mujoco.MjData,
                             camera_name: str) -> np.ndarray:
        """
        Transform points from camera frame to world frame.

        Args:
            points: (N, 3) points in camera frame
            data: MuJoCo data
            camera_name: Camera name

        Returns:
            (N, 3) points in world frame
        """
        cam_pos, cam_rot = self.get_camera_pose(data, camera_name)

        # Transform: p_world = R * p_camera + t
        points_world = (cam_rot @ points.T).T + cam_pos

        return points_world

    def world_to_camera_frame(self,
                             points: np.ndarray,
                             data: mujoco.MjData,
                             camera_name: str) -> np.ndarray:
        """
        Transform points from world frame to camera frame.

        Args:
            points: (N, 3) points in world frame
            data: MuJoCo data
            camera_name: Camera name

        Returns:
            (N, 3) points in camera frame
        """
        cam_pos, cam_rot = self.get_camera_pose(data, camera_name)

        # Transform: p_camera = R^T * (p_world - t)
        points_camera = (cam_rot.T @ (points - cam_pos).T).T

        return points_camera


class GraspPoseTransformer:
    """Transforms grasp poses between coordinate frames."""

    @staticmethod
    def grasp_to_ee_pose(grasp_pos: np.ndarray,
                        grasp_rot: np.ndarray,
                        approach_dist: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert grasp pose to end-effector target pose.

        Args:
            grasp_pos: (3,) grasp position
            grasp_rot: (3, 3) grasp orientation
            approach_dist: Distance to approach from (meters)

        Returns:
            Tuple of (ee_pos, ee_rot)
        """
        # Approach from above along grasp approach axis
        approach_vector = grasp_rot[:, 2]  # Z-axis of grasp frame
        ee_pos = grasp_pos - approach_dist * approach_vector
        ee_rot = grasp_rot.copy()

        return ee_pos, ee_rot

    @staticmethod
    def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.

        Args:
            quat: (4,) quaternion [w, x, y, z]

        Returns:
            (3, 3) rotation matrix
        """
        w, x, y, z = quat

        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])

    @staticmethod
    def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.

        Args:
            mat: (3, 3) rotation matrix

        Returns:
            (4,) quaternion [w, x, y, z]
        """
        trace = np.trace(mat)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (mat[2, 1] - mat[1, 2]) * s
            y = (mat[0, 2] - mat[2, 0]) * s
            z = (mat[1, 0] - mat[0, 1]) * s
        else:
            if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
                w = (mat[2, 1] - mat[1, 2]) / s
                x = 0.25 * s
                y = (mat[0, 1] + mat[1, 0]) / s
                z = (mat[0, 2] + mat[2, 0]) / s
            elif mat[1, 1] > mat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
                w = (mat[0, 2] - mat[2, 0]) / s
                x = (mat[0, 1] + mat[1, 0]) / s
                y = 0.25 * s
                z = (mat[1, 2] + mat[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
                w = (mat[1, 0] - mat[0, 1]) / s
                x = (mat[0, 2] + mat[2, 0]) / s
                y = (mat[1, 2] + mat[2, 1]) / s
                z = 0.25 * s

        return np.array([w, x, y, z])


def visualize_pointcloud(points: np.ndarray,
                        colors: np.ndarray,
                        grasp_poses: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None):
    """
    Visualize point cloud with optional grasp poses.

    Args:
        points: (N, 3) 3D points
        colors: (N, 3) RGB colors
        grasp_poses: Optional (M, 7) grasp poses [x, y, z, qw, qx, qy, qz]
        save_path: Optional path to save visualization
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Warning: open3d not installed. Cannot visualize point cloud.")
        print("Install with: pip install open3d")
        return

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Create coordinate frames for grasps
    geometries = [pcd]

    if grasp_poses is not None:
        for grasp in grasp_poses[:10]:  # Show top 10 grasps
            pos = grasp[:3]
            quat = grasp[3:7]
            rot = GraspPoseTransformer.quat_to_matrix(quat)

            # Create coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame.rotate(rot, center=(0, 0, 0))
            frame.translate(pos)
            geometries.append(frame)

    # Visualize
    o3d.visualization.draw_geometries(geometries)

    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Point cloud saved to {save_path}")


# Example usage
if __name__ == "__main__":
    print("Camera utilities module loaded successfully")
    print("\nAvailable classes:")
    print("  - CameraProcessor: Handle camera operations")
    print("  - GraspPoseTransformer: Transform grasp poses")
    print("\nAvailable functions:")
    print("  - visualize_pointcloud(): Visualize point clouds with Open3D")
