#!/usr/bin/env python3
"""
Visualization script for the hammer grasping environment.

This script loads and displays the hammer grasp scene with full physics simulation.
It allows real-time interaction with the MuJoCo viewer to test the environment setup
before implementing the RL training loop.

Controls:
  - Right mouse drag: Rotate camera
  - Scroll: Zoom camera
  - Space: Play/pause simulation
  - Right click on object: Apply force by dragging
  - 'C': Toggle camera tracking
"""

import mujoco
import mujoco.viewer
import numpy as np
import sys
import os

def main():
    """Load and visualize the hammer grasp scene."""

    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the XML file
    xml_path = os.path.join(script_dir, "hammer_grasp_rgbd_scene.xml")

    print(f"Loading scene from: {xml_path}")

    if not os.path.exists(xml_path):
        print(f"ERROR: Scene file not found at {xml_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Available files: {os.listdir(script_dir)}")
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
        print(f"  Timestep: {model.opt.timestep}s")

        # Print actuator names
        print(f"\nActuators ({model.nu} total):")
        for i in range(min(10, model.nu)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  {i}: {name}")
        if model.nu > 10:
            print(f"  ... and {model.nu - 10} more")

        # Print body names
        print(f"\nBodies (sample):")
        for i in [0, 1, model.nbody-2, model.nbody-1]:
            if i < model.nbody:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                print(f"  {i}: {name}")

        # Check for hammer and robot bodies
        print(f"\nKey bodies:")
        for body_name in ["hammer", "pelvis", "table", "right_hand_index_0_link"]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                print(f"  {body_name}: Found (id={body_id})")
            else:
                print(f"  {body_name}: NOT FOUND")

        # Create a viewer
        print("\nStarting MuJoCo viewer...")
        print("\nViewer Controls:")
        print("  Space: Play/pause")
        print("  Right mouse drag: Rotate view")
        print("  Scroll: Zoom")
        print("  'C': Toggle camera mode")
        print("  Click on scene: Apply forces")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Run simulation
            while viewer.is_running():
                # Step the simulation
                mujoco.mj_step(model, data)

                # Sync with viewer
                viewer.sync()

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
