#!/usr/bin/env python3
"""
Quick checker for hammer grasping environment scenes.
Verifies scene files are valid without requiring a display.
"""

import xml.etree.ElementTree as ET
import os

def check_scene(scene_file):
    """Check if a scene file is valid."""
    print(f"\nChecking: {scene_file}")
    print("=" * 60)

    if not os.path.exists(scene_file):
        print(f"[FAIL] File not found: {scene_file}")
        return False

    try:
        # Parse XML
        tree = ET.parse(scene_file)
        root = tree.getroot()
        print(f"[OK] XML parsed successfully")

        # Get model name
        model_name = root.get('model', 'unnamed')
        print(f"  Model: {model_name}")

        # Count elements
        bodies = root.findall('.//body')
        joints = root.findall('.//joint')
        geoms = root.findall('.//geom')
        meshes = root.findall('.//mesh')
        cameras = root.findall('.//camera')

        print(f"  Bodies: {len(bodies)}")
        print(f"  Joints: {len(joints)}")
        print(f"  Geoms: {len(geoms)}")
        print(f"  Meshes: {len(meshes)}")
        print(f"  Cameras: {len(cameras)}")

        # Check for fixed joint errors
        fixed_joints = [j for j in joints if j.get('type') == 'fixed']
        if fixed_joints:
            print(f"[FAIL] ERROR: Found {len(fixed_joints)} invalid 'fixed' joint types")
            return False
        else:
            print(f"[OK] No invalid 'fixed' joints")

        # Check for key bodies
        body_names = {b.get('name') for b in bodies}
        key_bodies = ['hammer', 'pelvis', 'table']
        for body in key_bodies:
            if body in body_names:
                print(f"[OK] Found body: {body}")
            else:
                print(f"[WARN] Missing body: {body}")

        # Check for mesh files
        mesh_refs = {}
        for mesh in meshes:
            name = mesh.get('name', 'unnamed')
            file = mesh.get('file', '')
            mesh_refs[name] = file

        print(f"\n  Checking {len(mesh_refs)} mesh references:")
        missing_meshes = []
        for name, filepath in mesh_refs.items():
            # Meshdir is 'assets' according to compiler tag
            full_path = os.path.join('assets', filepath)
            if os.path.exists(full_path):
                print(f"    [OK] {name}: {filepath}")
            else:
                print(f"    [FAIL] {name}: {filepath} (NOT FOUND)")
                missing_meshes.append(filepath)

        if missing_meshes:
            print(f"\n[FAIL] {len(missing_meshes)} mesh file(s) missing")
            return False
        else:
            print(f"\n[OK] All mesh files present")

        # Check for camera (RGBD scenes)
        if cameras:
            print(f"[OK] Scene has {len(cameras)} camera(s)")
            for i, cam in enumerate(cameras):
                cam_name = cam.get('name', f'camera_{i}')
                print(f"  - {cam_name}")

        print(f"\n[OK] Scene file is VALID")
        return True

    except ET.ParseError as e:
        print(f"[FAIL] XML Parse Error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

def main():
    """Check all scene files."""
    print("\n" + "=" * 60)
    print("HAMMER GRASPING ENVIRONMENT - SCENE CHECKER")
    print("=" * 60)

    scenes = [
        'hammer_grasp_scene.xml',
        'hammer_grasp_rgbd_scene.xml'
    ]

    results = {}
    for scene in scenes:
        results[scene] = check_scene(scene)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for scene, valid in results.items():
        status = "[OK] VALID" if valid else "[FAIL] INVALID"
        print(f"{status}: {scene}")

    all_valid = all(results.values())
    if all_valid:
        print("\n[OK] All scenes are valid and ready to use!")
        print("\nTo visualize:")
        print("  python visualize_hammer_grasp.py")
        return 0
    else:
        print("\n[FAIL] Some scenes have errors. Please fix them above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
