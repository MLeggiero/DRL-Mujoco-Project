#!/usr/bin/env python3
"""
Visualization script with live camera view for the hammer grasping environment.

This script shows both the main 3D viewer and a live camera feed in a separate window.
You can control the robot and see what the camera sees in real-time.

Controls:
  - Right mouse drag: Rotate camera in 3D view
  - Scroll: Zoom camera
  - Space: Play/pause simulation
  - Arrow keys: Control robot joints (when paused)
  - 'C': Cycle through cameras
  - 'ESC' or 'Q': Quit
"""

import mujoco
import mujoco.viewer
import numpy as np
import sys
import os
import cv2
import threading
import time

class CameraViewer:
    """Real-time camera viewer in a separate window."""

    def __init__(self, model, data, width=640, height=480):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.running = True

        # Create renderer
        self.renderer = mujoco.Renderer(model, height=height, width=width)

        # Get available cameras
        self.cameras = []
        for i in range(model.ncam):
            cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name and cam_name != 'track':  # Skip the tracking camera
                self.cameras.append(cam_name)

        if not self.cameras:
            print("No cameras found in the scene!")
            self.cameras = ['track']

        self.current_camera_idx = 0
        self.current_camera = self.cameras[0]

        print(f"Available cameras: {self.cameras}")
        print(f"Starting with camera: {self.current_camera}")

        # Window name
        self.window_name = "Robot Camera View (Press 'C' to cycle cameras, 'Q' to quit)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)

    def cycle_camera(self):
        """Switch to the next camera."""
        self.current_camera_idx = (self.current_camera_idx + 1) % len(self.cameras)
        self.current_camera = self.cameras[self.current_camera_idx]
        print(f"Switched to camera: {self.current_camera}")

    def capture_frame(self):
        """Capture a single frame from the current camera."""
        # Update scene with current camera
        self.renderer.update_scene(self.data, camera=self.current_camera)

        # Render RGB
        rgb = self.renderer.render()

        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Add camera name overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Camera: {self.current_camera}"
        cv2.putText(bgr, text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Add timestamp
        time_text = f"Time: {self.data.time:.2f}s"
        cv2.putText(bgr, time_text, (10, 60), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        return bgr

    def show_frame(self, frame):
        """Display the frame in the OpenCV window."""
        cv2.imshow(self.window_name, frame)

    def handle_input(self):
        """Handle keyboard input for the camera window."""
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            self.running = False
            return False
        elif key == ord('c') or key == ord('C'):
            self.cycle_camera()

        return True

    def close(self):
        """Close the camera viewer window."""
        cv2.destroyAllWindows()


def main():
    """Load and visualize the hammer grasp scene with camera view."""

    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the XML file
    xml_path = os.path.join(script_dir, "hammer_grasp_rgbd_scene.xml")

    print(f"Loading scene from: {xml_path}")

    if not os.path.exists(xml_path):
        print(f"ERROR: Scene file not found at {xml_path}")
        return 1

    try:
        # Load the model
        print("Loading MuJoCo model...")
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # Print model information
        print(f"\nModel Information:")
        print(f"  Bodies: {model.nbody}")
        print(f"  Joints: {model.njnt}")
        print(f"  Actuators: {model.nu}")
        print(f"  Cameras: {model.ncam}")
        print(f"  Timestep: {model.opt.timestep}s")

        # List all cameras
        print(f"\nAvailable cameras:")
        for i in range(model.ncam):
            cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            print(f"  {i}: {cam_name}")

        # Create camera viewer
        print("\nInitializing camera viewer...")
        camera_viewer = CameraViewer(model, data, width=640, height=480)

        print("\n=== Controls ===")
        print("3D Viewer:")
        print("  Space: Play/pause simulation")
        print("  Right mouse drag: Rotate view")
        print("  Scroll: Zoom")
        print("  Double-click: Select body")
        print("\nCamera Window:")
        print("  C: Cycle through cameras")
        print("  Q or ESC: Quit")
        print("================\n")

        # Launch the passive viewer (non-blocking)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Main simulation loop
            while viewer.is_running() and camera_viewer.running:
                # Step the simulation
                mujoco.mj_step(model, data)

                # Sync with 3D viewer
                viewer.sync()

                # Capture and display camera frame
                frame = camera_viewer.capture_frame()
                camera_viewer.show_frame(frame)

                # Handle camera viewer input
                if not camera_viewer.handle_input():
                    break

                # Small sleep to prevent overwhelming the CPU
                time.sleep(0.001)

        # Clean up
        camera_viewer.close()
        print("\nViewer closed. Simulation complete.")
        return 0

    except Exception as e:
        print(f"ERROR: Failed to load or visualize scene")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
