#!/usr/bin/env python3
"""
Complete VLM-based grasping demonstration.

This script demonstrates the full pipeline:
1. Capture RGB-D from MuJoCo camera
2. Generate point cloud
3. Detect grasps with VLM (or heuristic fallback)
4. Plan motion with IK
5. Execute and visualize

Usage:
    python3 vlm_grasp_demo.py --visualize --num-attempts 10
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # Headless rendering

import numpy as np
import mujoco
import mujoco.viewer
import argparse
import time
from pathlib import Path

# Import our modules
from camera_utils import CameraProcessor
from grasp_detector import create_grasp_detector, Grasp
from motion_planner import MotionPlanner


class VLMGraspingSystem:
    """Complete VLM-based grasping system."""

    def __init__(self,
                 scene_path: str = "hammer_grasp_rgbd_scene.xml",
                 grasp_backend: str = 'heuristic',
                 camera_name: str = 'track_front',
                 image_size: tuple = (640, 480),
                 save_pointclouds: bool = True,
                 output_dir: str = "./pointcloud_data"):
        """
        Initialize VLM grasping system.

        Args:
            scene_path: Path to MuJoCo scene XML
            grasp_backend: Grasp detector backend ('anygrasp', 'graspnet', 'heuristic', 'auto')
            camera_name: Camera to use for perception
            image_size: (width, height) for rendering
            save_pointclouds: Whether to save point clouds to disk
            output_dir: Directory to save point clouds (overwritten each run)
        """
        self.save_pointclouds = save_pointclouds
        self.output_dir = output_dir

        # Create output directory
        if self.save_pointclouds:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Point clouds will be saved to: {self.output_dir}")
        # Load MuJoCo model
        if not os.path.exists(scene_path):
            scene_path = os.path.join(os.path.dirname(__file__), scene_path)

        print(f"Loading scene: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        print(f"✓ Model loaded: {self.model.nbody} bodies, {self.model.njnt} joints")

        # Initialize components
        self.camera_name = camera_name
        self.image_size = image_size

        print(f"\n Initializing vision system...")
        self.camera_processor = CameraProcessor(self.model, width=image_size[0], height=image_size[1])
        self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])

        print(f"Initializing grasp detector ({grasp_backend})...")
        self.grasp_detector = create_grasp_detector(grasp_backend)

        print(f"✓ Initializing motion planner...")
        self.motion_planner = MotionPlanner(self.model, self.data)

        # Lock legs for stability
        self._lock_legs()

        print("✓ System ready!\n")

    def _lock_legs(self):
        """Lock leg joints for stable base."""
        leg_joints = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]

        for joint_name in leg_joints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                # Set joint to neutral position
                self.data.qpos[joint_id] = 0.0

    def capture_rgbd(self):
        """Capture RGB-D from camera using proper MuJoCo depth rendering."""
        # Need to step forward first
        mujoco.mj_forward(self.model, self.data)

        # Create a scene and context for depth rendering (using old API that works)
        scene = mujoco.MjvScene(self.model, maxgeom=10000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Set up camera
        camera = mujoco.MjvCamera()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = cam_id

        # Create viewport
        viewport = mujoco.MjrRect(0, 0, self.image_size[0], self.image_size[1])

        # Update scene
        mujoco.mjv_updateScene(
            self.model, self.data,
            mujoco.MjvOption(), None, camera,
            mujoco.mjtCatBit.mjCAT_ALL, scene
        )

        # Render RGB
        mujoco.mjr_render(viewport, scene, context)
        rgb = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        depth_raw = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth_raw, viewport, context)

        # Flip RGB vertically (OpenGL convention)
        rgb = np.flipud(rgb)

        # Convert depth from OpenGL NDC to actual distance
        # MuJoCo depth is in normalized device coordinates [-1, 1]
        # We need to convert to actual world distances
        depth = self._ndc_to_depth(depth_raw, cam_id)

        # Flip depth vertically
        depth = np.flipud(depth)

        return rgb, depth

    def _ndc_to_depth(self, ndc_depth, cam_id):
        """Convert NDC depth to actual distance in meters.

        Standard OpenGL depth buffer conversion formula.
        MuJoCo uses perspective projection with depth buffer in [0, 1].
        """
        # Get camera near and far planes
        extent = self.model.stat.extent

        # MuJoCo's default near/far planes
        near = 0.01
        far = extent * 5

        # Clip to valid range [0, 1]
        ndc_depth = np.clip(ndc_depth, 0, 1)

        # Standard perspective depth conversion from NDC to linear depth
        # This is the inverse of the perspective projection:
        # z_ndc = (far + near) / (far - near) - (2 * far * near) / (z * (far - near))
        #
        # Solving for z (actual depth):
        # z = (2 * far * near) / (far + near - z_ndc * (far - near))
        #
        # Since MuJoCo stores depth in [0,1] not [-1,1], we use:
        z_ndc = 2.0 * ndc_depth - 1.0  # Map [0,1] to [-1,1]

        depth = (2.0 * far * near) / (far + near - z_ndc * (far - near))

        return depth

    def _create_depth_from_scene(self):
        """Create depth map from scene geometry (workaround)."""
        # Get camera pose
        cam_pos, cam_rot = self.camera_processor.get_camera_pose(self.data, self.camera_name)

        # Get hammer position
        hammer_pos = self.data.body('hammer').xpos

        # Distance from camera to hammer
        dist_to_hammer = np.linalg.norm(hammer_pos - cam_pos)

        # Create a simple depth map with hammer at center
        depth = np.ones((self.image_size[1], self.image_size[0]), dtype=np.float32) * 10.0

        # Place hammer depth at center
        cy, cx = self.image_size[1] // 2, self.image_size[0] // 2
        radius = 50  # pixels

        y, x = np.ogrid[:self.image_size[1], :self.image_size[0]]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        depth[mask] = dist_to_hammer

        # Add some table depth
        depth[self.image_size[1]//2:] = min(dist_to_hammer + 0.2, 5.0)

        return depth

    def detect_grasps(self, num_grasps: int = 10, attempt_num: int = 0):
        """
        Detect grasps from current scene.

        Args:
            num_grasps: Number of grasp candidates to return
            attempt_num: Attempt number for filename

        Returns:
            List of Grasp objects
        """
        # Capture RGB-D
        rgb, depth = self.capture_rgbd()

        # Convert to point cloud
        points, colors = self.camera_processor.rgbd_to_pointcloud(
            rgb, depth, self.camera_name
        )

        print(f"Point cloud: {len(points)} points")

        # Transform to world frame
        points_world = self.camera_processor.camera_to_world_frame(
            points, self.data, self.camera_name
        )

        # Save point cloud and RGB-D
        if self.save_pointclouds and len(points_world) > 0:
            self._save_capture(rgb, depth, points_world, colors, attempt_num)

        # Detect grasps
        grasps = self.grasp_detector.detect(points_world, colors, num_grasps=num_grasps)

        print(f"Detected {len(grasps)} grasp candidates")

        return grasps

    def _save_capture(self, rgb, depth, points, colors, attempt_num):
        """Save RGB, depth, and point cloud to disk."""
        import matplotlib.pyplot as plt

        # Save RGB image (overwrite)
        rgb_path = os.path.join(self.output_dir, "rgb.png")
        plt.imsave(rgb_path, rgb)

        # Save depth image (overwrite)
        depth_path = os.path.join(self.output_dir, "depth.png")
        plt.imsave(depth_path, depth, cmap='viridis')

        # Save point cloud as numpy array (overwrite)
        pcd_path = os.path.join(self.output_dir, "pointcloud.npz")
        np.savez(pcd_path, points=points, colors=colors)

        # Save point cloud as PLY for visualization (overwrite)
        ply_path = os.path.join(self.output_dir, "pointcloud.ply")
        self._save_ply(points, colors, ply_path)

        print(f"  Saved: RGB, depth, and point cloud to {self.output_dir}")

    def _save_ply(self, points, colors, filename):
        """Save point cloud as PLY file."""
        with open(filename, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write points
            for point, color in zip(points, colors):
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
                f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

    def execute_grasp(self, grasp: Grasp):
        """
        Execute a grasp attempt.

        Args:
            grasp: Grasp object to execute

        Returns:
            True if successful, False otherwise
        """
        print(f"\nExecuting grasp: pos={grasp.position}, score={grasp.score:.3f}")

        # Plan motion
        print("  Planning trajectory...")
        trajectory, success = self.motion_planner.plan_to_grasp(
            grasp.position,
            grasp.quaternion,
            approach_distance=0.15,
            approach_steps=50,
            grasp_steps=30
        )

        if not success:
            print("  ✗ IK failed")
            return False

        print(f"  ✓ Planned {len(trajectory)} waypoints")

        # Execute trajectory
        print("  Executing motion...")
        for i, waypoint in enumerate(trajectory):
            self.data.qpos[:] = waypoint
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

            # Yield for visualization
            if i % 10 == 0:
                time.sleep(0.01)

        # Check if we're near the hammer
        hammer_pos = self.data.body('hammer').xpos
        hand_pos = self.data.site('right_palm').xpos
        distance = np.linalg.norm(hammer_pos - hand_pos)

        print(f"  Final distance to hammer: {distance:.3f}m")

        # Success if within 10cm
        success = distance < 0.10

        if success:
            print("  ✓ Grasp successful!")
        else:
            print("  ✗ Grasp failed (too far from object)")

        return success

    def run_demo(self, num_attempts: int = 5, visualize: bool = True):
        """
        Run grasping demo with multiple attempts.

        Args:
            num_attempts: Number of grasp attempts
            visualize: Whether to open visualization

        Returns:
            Success rate
        """
        print("="*60)
        print("VLM GRASPING DEMONSTRATION")
        print("="*60)

        successes = 0

        for attempt in range(num_attempts):
            print(f"\n--- Attempt {attempt + 1}/{num_attempts} ---")

            # Reset scene
            mujoco.mj_resetData(self.model, self.data)
            self._lock_legs()

            # Detect grasps
            grasps = self.detect_grasps(num_grasps=10, attempt_num=attempt)

            if len(grasps) == 0:
                print("No grasps detected!")
                continue

            # Try top 3 grasps
            grasp_success = False
            for i, grasp in enumerate(grasps[:3]):
                print(f"\nTrying grasp {i+1}/3 (score={grasp.score:.3f})")

                success = self.execute_grasp(grasp)

                if success:
                    successes += 1
                    grasp_success = True
                    break

            if not grasp_success:
                print("\n✗ All grasps failed for this attempt")

        # Summary
        success_rate = successes / num_attempts * 100
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Attempts: {num_attempts}")
        print(f"Successes: {successes}")
        print(f"Success rate: {success_rate:.1f}%")
        print("="*60)

        return success_rate


def run_with_visualization(system: VLMGraspingSystem):
    """Run system with interactive visualization."""
    print("\nLaunching interactive visualization...")
    print("Controls:")
    print("  Space: Pause/Resume")
    print("  G: Detect and execute grasp")
    print("  R: Reset scene")
    print("  ESC: Quit")

    with mujoco.viewer.launch_passive(system.model, system.data) as viewer:
        step_count = 0

        while viewer.is_running():
            # Step simulation
            mujoco.mj_step(system.model, system.data)
            viewer.sync()

            step_count += 1

            # Auto-detect grasps every 500 steps
            if step_count % 500 == 0:
                print("\n[Auto] Detecting grasps...")
                grasps = system.detect_grasps(num_grasps=5)

                if len(grasps) > 0:
                    print(f"[Auto] Executing best grasp (score={grasps[0].score:.3f})...")
                    system.execute_grasp(grasps[0])

            time.sleep(0.001)


def main():
    parser = argparse.ArgumentParser(
        description='VLM-based grasp detection and execution demo',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--backend', type=str, default='heuristic',
                       choices=['auto', 'anygrasp', 'graspnet', 'heuristic'],
                       help='Grasp detection backend')
    parser.add_argument('--camera', type=str, default='track_front',
                       help='Camera name to use')
    parser.add_argument('--num-attempts', type=int, default=5,
                       help='Number of grasp attempts')
    parser.add_argument('--visualize', action='store_true',
                       help='Open interactive visualization')
    parser.add_argument('--headless', action='store_true',
                       help='Run without visualization (benchmark mode)')

    args = parser.parse_args()

    # Create system
    system = VLMGraspingSystem(
        grasp_backend=args.backend,
        camera_name=args.camera
    )

    if args.visualize and not args.headless:
        # Interactive mode
        run_with_visualization(system)
    else:
        # Benchmark mode
        system.run_demo(num_attempts=args.num_attempts, visualize=False)


if __name__ == "__main__":
    main()
