#!/usr/bin/env python3
"""
Camera data capture script for the hammer grasping environment.

This script renders camera images from the RGBD scene and saves them to disk.
Supports RGB, depth, and combined RGBD outputs from multiple cameras.

Usage:
    python3 capture_camera_data.py --camera track_left --frames 100 --output ./camera_data
    python3 capture_camera_data.py --all-cameras --frames 50 --fps 30
"""

import mujoco
import numpy as np
import sys
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


class CameraDataCollector:
    """Collects visual data from MuJoCo cameras."""

    def __init__(self, model, data, width=640, height=480):
        """
        Initialize the camera data collector.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            width: Image width in pixels
            height: Image height in pixels
        """
        self.model = model
        self.data = data
        self.width = width
        self.height = height

        # Create renderer
        self.renderer = mujoco.Renderer(model, height=height, width=width)

        # List available cameras
        self.cameras = []
        for i in range(model.ncam):
            cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name:
                self.cameras.append(cam_name)

        print(f"Available cameras: {self.cameras}")

    def get_camera_id(self, camera_name):
        """Get camera ID from name."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found. Available: {self.cameras}")
        return cam_id

    def capture_rgb(self, camera_name):
        """
        Capture RGB image from specified camera.

        Args:
            camera_name: Name of the camera to render from

        Returns:
            RGB image as numpy array (height, width, 3) with values 0-255
        """
        # Update renderer to use the specified camera
        self.renderer.update_scene(self.data, camera=camera_name)

        # Render RGB
        rgb = self.renderer.render()

        return rgb

    def capture_depth(self, camera_name):
        """
        Capture depth image from specified camera.

        Args:
            camera_name: Name of the camera to render from

        Returns:
            Depth image as numpy array (height, width) with values in meters
        """
        # Update scene
        self.renderer.update_scene(self.data, camera=camera_name)

        # Enable depth rendering
        self.renderer.enable_depth_rendering()

        # Render depth
        depth = self.renderer.render()

        # Disable depth rendering for next RGB render
        self.renderer.disable_depth_rendering()

        return depth

    def capture_rgbd(self, camera_name):
        """
        Capture both RGB and depth from specified camera.

        Args:
            camera_name: Name of the camera to render from

        Returns:
            Tuple of (rgb, depth) as numpy arrays
        """
        rgb = self.capture_rgb(camera_name)
        depth = self.capture_depth(camera_name)

        return rgb, depth

    def save_rgb(self, rgb, filepath):
        """Save RGB image to file."""
        plt.imsave(filepath, rgb)

    def save_depth(self, depth, filepath, colormap='viridis'):
        """Save depth image to file with colormap."""
        plt.imsave(filepath, depth, cmap=colormap)

    def save_depth_raw(self, depth, filepath):
        """Save raw depth values as numpy array."""
        np.save(filepath, depth)

    def save_rgbd_combined(self, rgb, depth, filepath):
        """Save RGB and depth side-by-side visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(rgb)
        axes[0].set_title('RGB')
        axes[0].axis('off')

        depth_img = axes[1].imshow(depth, cmap='viridis')
        axes[1].set_title('Depth (meters)')
        axes[1].axis('off')

        plt.colorbar(depth_img, ax=axes[1])
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()


def run_data_collection(args):
    """Run the data collection process."""

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "hammer_grasp_rgbd_scene.xml")

    if not os.path.exists(xml_path):
        print(f"ERROR: Scene file not found at {xml_path}")
        return 1

    print(f"Loading scene from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Initialize collector
    collector = CameraDataCollector(model, data, width=args.width, height=args.height)

    # Determine which cameras to use
    if args.all_cameras:
        cameras_to_use = collector.cameras
    else:
        cameras_to_use = [args.camera]

    print(f"Capturing from cameras: {cameras_to_use}")
    print(f"Frames to capture: {args.frames}")
    print(f"Resolution: {args.width}x{args.height}")

    # Simulation parameters
    dt = model.opt.timestep
    fps = args.fps
    frame_interval = int(1.0 / (fps * dt))  # Steps between frames

    print(f"Simulation timestep: {dt}s")
    print(f"Target FPS: {fps}")
    print(f"Capturing every {frame_interval} steps")

    # Create subdirectories for each camera
    camera_dirs = {}
    for cam_name in cameras_to_use:
        cam_dir = output_dir / cam_name
        cam_dir.mkdir(exist_ok=True)
        camera_dirs[cam_name] = cam_dir

        # Create subdirs for different data types
        (cam_dir / "rgb").mkdir(exist_ok=True)
        (cam_dir / "depth").mkdir(exist_ok=True)
        (cam_dir / "depth_raw").mkdir(exist_ok=True)
        if args.save_combined:
            (cam_dir / "rgbd_combined").mkdir(exist_ok=True)

    # Run simulation and capture frames
    frame_count = 0
    step_count = 0

    print("\nStarting data collection...")

    try:
        while frame_count < args.frames:
            # Step simulation
            mujoco.mj_step(model, data)
            step_count += 1

            # Capture frame at intervals
            if step_count % frame_interval == 0:
                for cam_name in cameras_to_use:
                    cam_dir = camera_dirs[cam_name]

                    # Capture RGBD
                    rgb, depth = collector.capture_rgbd(cam_name)

                    # Save files
                    frame_id = f"{frame_count:06d}"

                    if args.save_rgb:
                        rgb_path = cam_dir / "rgb" / f"rgb_{frame_id}.png"
                        collector.save_rgb(rgb, rgb_path)

                    if args.save_depth:
                        depth_path = cam_dir / "depth" / f"depth_{frame_id}.png"
                        collector.save_depth(depth, depth_path)

                    if args.save_depth_raw:
                        depth_raw_path = cam_dir / "depth_raw" / f"depth_{frame_id}.npy"
                        collector.save_depth_raw(depth, depth_raw_path)

                    if args.save_combined:
                        combined_path = cam_dir / "rgbd_combined" / f"rgbd_{frame_id}.png"
                        collector.save_rgbd_combined(rgb, depth, combined_path)

                frame_count += 1

                if frame_count % 10 == 0:
                    print(f"Progress: {frame_count}/{args.frames} frames captured")

        print(f"\nData collection complete!")
        print(f"Captured {frame_count} frames from {len(cameras_to_use)} camera(s)")
        print(f"Total simulation steps: {step_count}")
        print(f"Simulation time: {step_count * dt:.2f}s")
        print(f"Output saved to: {output_dir}")

        # Save metadata
        metadata = {
            'frames': frame_count,
            'cameras': cameras_to_use,
            'resolution': f"{args.width}x{args.height}",
            'fps': fps,
            'timestep': dt,
            'total_steps': step_count,
            'sim_time': step_count * dt,
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = output_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

        print(f"Metadata saved to: {metadata_path}")

        return 0

    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user")
        print(f"Captured {frame_count} frames before interruption")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Capture camera data from hammer grasping environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture 100 frames from left camera at 30 FPS
  python3 capture_camera_data.py --camera track_left --frames 100 --fps 30

  # Capture from all cameras at 640x480 resolution
  python3 capture_camera_data.py --all-cameras --frames 50 --width 640 --height 480

  # Capture with custom output directory
  python3 capture_camera_data.py --camera track_front --frames 200 --output ./my_data

  # Capture only RGB images (no depth)
  python3 capture_camera_data.py --camera track_left --no-depth --frames 100
        """
    )

    parser.add_argument('--camera', type=str, default='track_left',
                        help='Camera name to capture from (default: track_left)')
    parser.add_argument('--all-cameras', action='store_true',
                        help='Capture from all available cameras')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames to capture (default: 100)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target frames per second (default: 30)')
    parser.add_argument('--width', type=int, default=640,
                        help='Image width in pixels (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Image height in pixels (default: 480)')
    parser.add_argument('--output', type=str, default='./camera_data',
                        help='Output directory for captured data (default: ./camera_data)')

    # Data type options
    parser.add_argument('--no-rgb', dest='save_rgb', action='store_false',
                        help='Do not save RGB images')
    parser.add_argument('--no-depth', dest='save_depth', action='store_false',
                        help='Do not save depth images (colormap)')
    parser.add_argument('--no-depth-raw', dest='save_depth_raw', action='store_false',
                        help='Do not save raw depth arrays (.npy)')
    parser.add_argument('--save-combined', action='store_true',
                        help='Save combined RGB+depth visualization')

    parser.set_defaults(save_rgb=True, save_depth=True, save_depth_raw=True)

    args = parser.parse_args()

    return run_data_collection(args)


if __name__ == "__main__":
    sys.exit(main())
