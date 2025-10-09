#!/usr/bin/env python3
"""
Custom MuJoCo Scene Launcher for Unitree G1.
This version uses a simplified path structure: 
The scene XML (g1_table_box_scene.xml) is expected to be inside the unitree_g1/ folder.
"""
import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import time

# --- File Path Constants ---
# The scene file is now inside the unitree_g1 folder.
G1_ROOT_DIR = "unitree_g1"
TABLE_BOX_SCENE_PATH = os.path.join(G1_ROOT_DIR, "g1_table_box_scene.xml")

def setup_model_for_viewing(model, data, robot_position=None):
    """Set up model in optimal viewing state for Unitree G1."""
    
    print("Setting up Unitree G1 for viewing...")
    
    # Reset velocities
    data.qvel[:] = 0
    
    # Default robot position (G1 standing height)
    if robot_position is None:
        robot_position = [-0.1, 0.0, 0.8]  # [x, y, z] 
    
    print(f"Setting G1 position to: [{robot_position[0]:.2f}, {robot_position[1]:.2f}, {robot_position[2]:.2f}]")
    
    # Set reasonable joint positions for G1 humanoid
    if model.njnt > 0:
        for i in range(model.njnt):
            joint_type = model.jnt_type[i]
            qpos_addr = model.jnt_qposadr[i]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            
            if joint_type == mujoco.mjtJoint.mjJNT_FREE and qpos_addr + 6 < model.nq:
                # Free joint: standing position for G1
                data.qpos[qpos_addr:qpos_addr+3] = robot_position  # Position
                data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]  # Quaternion (w,x,y,z)
            
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE and qpos_addr < model.nq:
                # Hinge: G1-specific joint positions for a standing pose
                joint_lower = joint_name.lower()
                
                if 'knee' in joint_lower:
                    data.qpos[qpos_addr] = 0.2    # Slight knee bend
                elif 'hip' in joint_lower and 'pitch' in joint_lower:
                    data.qpos[qpos_addr] = -0.15  # Hip pitch for standing
                elif 'elbow' in joint_lower:
                    data.qpos[qpos_addr] = -0.3   # Natural elbow bend
                else:
                    data.qpos[qpos_addr] = 0.0    # Neutral for other joints
            
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL and qpos_addr + 3 < model.nq:
                # Ball joint: neutral orientation
                data.qpos[qpos_addr:qpos_addr+4] = [1, 0, 0, 0]
    
    # Update physics
    mujoco.mj_forward(model, data)
    print("G1 model setup complete")

def test_model_loading(model_path):
    """Test if a model can be loaded and catch MuJoCo errors."""
    try:
        # NOTE: from_xml_path() automatically adds the directory of the model file 
        # to the asset search path, which fixes the previous inclusion errors.
        model = mujoco.MjModel.from_xml_path(model_path)
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def view_custom_g1_scene(model_path, robot_position=None):
    """View a custom G1 scene with enhanced setup."""
    print(f"\nLoading custom Unitree G1 scene: {model_path}")
    print("-" * 60)
    
    # Test loading first
    test_result = test_model_loading(model_path)
    if not test_result['success']:
        print(f"Error loading model: {test_result.get('error', 'Unknown Error')}")
        return False
    
    try:
        # Load model
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        # Setup G1 for optimal viewing with custom position
        setup_model_for_viewing(model, data, robot_position)
        
        # Display controls
        print(f"\nUNITREE G1 VIEWER CONTROLS")
        print("-" * 40)
        print("ESC: Close viewer | Space: Pause/resume")
        print("Mouse drag: Rotate camera | Shift+Mouse: Pan")
        print("-" * 60)
        
        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            last_reset = time.time()
            step_count = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # Auto-reset every 25 seconds (optional)
                if time.time() - last_reset > 25:
                    print("Auto-resetting G1 pose...")
                    setup_model_for_viewing(model, data, robot_position)
                    last_reset = time.time()
                
                # Step physics
                mujoco.mj_step(model, data)
                viewer.sync()
                step_count += 1
                
                # Real-time control (sleep to maintain simulation speed)
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print(f"Custom G1 scene viewer closed")
        return True
        
    except Exception as e:
        print(f"Error viewing G1 scene: {e}")
        return False

def main():
    """Main function to launch the scene."""
    
    print("Custom Unitree G1 Scene Launcher (Simplified Path Fix)")
    print("=" * 60)

    # 1. Check for the G1 XML file and its assets
    g1_xml_path = os.path.join(G1_ROOT_DIR, "g1.xml")
    g1_asset_dir = os.path.join(G1_ROOT_DIR, "assets")

    if not os.path.exists(g1_xml_path) or not os.path.exists(g1_asset_dir):
        print("--- FATAL SETUP ERROR ---")
        print(f"Required Unitree G1 model files not found. The directory structure MUST be:")
        print(f"  {os.getcwd()}")
        print(f"  └── {G1_ROOT_DIR}/")
        print(f"      ├── g1.xml")
        print(f"      └── assets/ (contains STL files)")
        sys.exit(1)
        
    # 2. Check if the scene XML file exists in its new, fixed location
    if not os.path.exists(TABLE_BOX_SCENE_PATH):
        print(f"FATAL ERROR: Scene file not found at '{TABLE_BOX_SCENE_PATH}'")
        print(f"Please create/move 'g1_table_box_scene.xml' into the '{G1_ROOT_DIR}' folder.")
        sys.exit(1)

    # 3. Launch the requested scene
    print(f"Attempting to launch the required scene: '{TABLE_BOX_SCENE_PATH}'")
    view_custom_g1_scene(TABLE_BOX_SCENE_PATH)


if __name__ == "__main__":
    main()