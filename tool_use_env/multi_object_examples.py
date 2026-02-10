#!/usr/bin/env python3
"""
Practical examples for multi-object detection.

Shows how to:
1. Detect multiple tools at once
2. Track robot hands/grippers
3. Separate tools from hands
4. Get 3D positions for all objects
5. Use in RL environment
"""

import numpy as np
import cv2
from multi_object_detector import MultiObjectDetector


def example_1_detect_multiple_tools():
    """Example 1: Detect multiple tool types in one image."""
    print("\n" + "="*60)
    print("Example 1: Detect Multiple Tools")
    print("="*60)

    detector = MultiObjectDetector(verbose=False)
    rgb = cv2.imread("pointcloud_data/rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Detect multiple tools at once
    tools = ["hammer", "screwdriver", "wrench", "pliers"]

    detections = detector.detect_multiple_objects(
        rgb,
        prompts=tools,
        box_threshold=0.25,
        combine_results=True  # Combine all into one list
    )

    print(f"Detected {len(detections)} objects")
    for i, det in enumerate(detections):
        print(f"\n  Detection {i+1}:")
        print(f"    Type: {det['label']}")
        print(f"    Confidence: {det['confidence']:.1%}")
        print(f"    Position: {det['center']}")
        print(f"    Size: {det['geometry']['width']}x{det['geometry']['height']} px")
        print(f"    Orientation: {det['geometry']['orientation']}")

    return detections


def example_2_separate_detection():
    """Example 2: Detect each tool type separately."""
    print("\n" + "="*60)
    print("Example 2: Separate Detection by Type")
    print("="*60)

    detector = MultiObjectDetector(verbose=False)
    rgb = cv2.imread("pointcloud_data/rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Detect each type separately
    tools = ["hammer", "screwdriver", "wrench"]

    detections_by_type = detector.detect_multiple_objects(
        rgb,
        prompts=tools,
        box_threshold=0.25,
        combine_results=False  # Separate dict for each type
    )

    for tool_type, detections in detections_by_type.items():
        print(f"\n{tool_type.upper()}:")
        if len(detections) > 0:
            for det in detections:
                print(f"  ✓ Found at {det['center']} ({det['confidence']:.1%})")
        else:
            print(f"  ✗ Not found")

    return detections_by_type


def example_3_detect_hands_and_tools():
    """Example 3: Detect both robot hands and tools."""
    print("\n" + "="*60)
    print("Example 3: Detect Hands and Tools")
    print("="*60)

    detector = MultiObjectDetector(verbose=False)
    rgb = cv2.imread("pointcloud_data/rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Method 1: Use convenience functions
    print("\nMethod 1: Using convenience functions")
    tools = detector.detect_tools(rgb, specific_tools=["hammer", "wrench"], box_threshold=0.30)
    hands = detector.detect_hands(rgb, box_threshold=0.25)

    print(f"  Tools: {len(tools)} detected")
    for tool in tools:
        print(f"    - {tool['label']}: {tool['confidence']:.1%}")

    print(f"  Hands: {len(hands)} detected")
    for hand in hands[:3]:  # Show top 3
        print(f"    - {hand['label']}: {hand['confidence']:.1%}")

    # Method 2: Use complete scene detection
    print("\nMethod 2: Complete scene detection")
    scene = detector.detect_scene(
        rgb,
        include_tools=True,
        include_hands=True,
        include_objects=False,
        tool_threshold=0.30,
        hand_threshold=0.25
    )

    summary = detector.summarize_scene(scene)
    print(f"  Total objects: {summary['total_objects']}")
    print(f"  By category: {summary['counts']}")

    return scene


def example_4_filter_by_geometry():
    """Example 4: Use geometry filtering to separate tools from hands."""
    print("\n" + "="*60)
    print("Example 4: Geometry-Based Filtering")
    print("="*60)

    detector = MultiObjectDetector(verbose=False)
    rgb = cv2.imread("pointcloud_data/rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Detect everything with generic prompts
    all_detections = detector.detect(
        rgb,
        text_prompt="tool . robot hand . gripper . hammer",
        box_threshold=0.25
    )

    print(f"Total detections: {len(all_detections)}")

    # Filter by category using geometry
    tools = detector.filter_by_category(all_detections, 'tool', rgb.shape)
    hands = detector.filter_by_category(all_detections, 'hand', rgb.shape)

    print(f"\nAfter geometry filtering:")
    print(f"  Tools: {len(tools)}")
    for tool in tools:
        geom = tool['geometry']
        print(f"    - {tool['label']}: AR={geom['aspect_ratio']:.2f}, {geom['orientation']}")

    print(f"  Hands: {len(hands)}")
    for hand in hands[:3]:
        geom = hand['geometry']
        print(f"    - {hand['label']}: AR={geom['aspect_ratio']:.2f}, {geom['orientation']}")

    return tools, hands


def example_5_3d_positions():
    """Example 5: Get 3D positions for all detected objects."""
    print("\n" + "="*60)
    print("Example 5: 3D Position Estimation")
    print("="*60)

    detector = MultiObjectDetector(verbose=False)

    # Load RGB and depth
    rgb = cv2.imread("pointcloud_data/rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth_img = cv2.imread("pointcloud_data/depth.png", cv2.IMREAD_UNCHANGED)
    if len(depth_img.shape) == 3:
        depth_img = depth_img[:, :, 0]
    depth = depth_img.astype(np.float32) / 255.0 * 1.5 + 0.5  # Scale to meters

    # Camera intrinsics (example)
    K = np.array([
        [525.0, 0, 320.0],
        [0, 525.0, 240.0],
        [0, 0, 1.0]
    ])

    # Detect scene
    scene = detector.detect_scene(
        rgb,
        include_tools=True,
        include_hands=True,
        tool_threshold=0.30,
        hand_threshold=0.30
    )

    # Get 3D positions
    print("\n3D Positions:")

    for category, detections in scene.items():
        if len(detections) == 0:
            continue

        print(f"\n{category.upper()}:")
        for i, det in enumerate(detections[:3]):  # Top 3 per category
            pos_3d = detector.get_3d_position(det, depth, K)
            print(f"  {i+1}. {det['label']}")
            print(f"     2D: {det['center']} px")
            print(f"     3D: [{pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f}] m")
            print(f"     Confidence: {det['confidence']:.1%}")

    return scene


def example_6_visualize_categories():
    """Example 6: Visualize with color-coded categories."""
    print("\n" + "="*60)
    print("Example 6: Color-Coded Visualization")
    print("="*60)

    detector = MultiObjectDetector(verbose=False)
    rgb = cv2.imread("pointcloud_data/rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Detect complete scene
    scene = detector.detect_scene(
        rgb,
        include_tools=True,
        include_hands=True,
        tool_threshold=0.30,
        hand_threshold=0.25
    )

    # Visualize with color coding
    # Green = tools, Red = hands, Blue = objects
    vis_img = detector.visualize_detections(
        rgb,
        scene,
        save_path="scene_detection_colored.png",
        show_geometry=True
    )

    print("\n✓ Visualization saved to: scene_detection_colored.png")
    print("  Green boxes = Tools")
    print("  Red boxes = Robot hands/grippers")
    print("  Blue boxes = Objects")

    return vis_img


def example_7_track_specific_hand():
    """Example 7: Track a specific robot hand (left or right)."""
    print("\n" + "="*60)
    print("Example 7: Track Specific Hand")
    print("="*60)

    detector = MultiObjectDetector(verbose=False)
    rgb = cv2.imread("pointcloud_data/rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Detect both hands
    left_hand = detector.detect(rgb, text_prompt="left robot hand", box_threshold=0.25)
    right_hand = detector.detect(rgb, text_prompt="right robot hand", box_threshold=0.25)

    print(f"\nLeft hand detections: {len(left_hand)}")
    if len(left_hand) > 0:
        best = left_hand[0]
        print(f"  Best: {best['label']} at {best['center']} ({best['confidence']:.1%})")

    print(f"\nRight hand detections: {len(right_hand)}")
    if len(right_hand) > 0:
        best = right_hand[0]
        print(f"  Best: {best['label']} at {best['center']} ({best['confidence']:.1%})")

    # Or detect all hands and separate by position
    all_hands = detector.detect_hands(rgb, box_threshold=0.25)

    # Split by x-position (left half vs right half of image)
    image_center_x = rgb.shape[1] // 2

    left_hands = [h for h in all_hands if h['center'][0] < image_center_x]
    right_hands = [h for h in all_hands if h['center'][0] >= image_center_x]

    print(f"\nSeparated by position:")
    print(f"  Left side: {len(left_hands)} detections")
    print(f"  Right side: {len(right_hands)} detections")

    return left_hands, right_hands


def example_8_workspace_analysis():
    """Example 8: Analyze workspace - find reachable tools."""
    print("\n" + "="*60)
    print("Example 8: Workspace Analysis")
    print("="*60)

    detector = MultiObjectDetector(verbose=False)
    rgb = cv2.imread("pointcloud_data/rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Detect scene
    scene = detector.detect_scene(rgb, tool_threshold=0.30, hand_threshold=0.25)

    # Get hand positions (as gripper locations)
    hands = scene['hands']
    tools = scene['tools']

    if len(hands) > 0 and len(tools) > 0:
        # Calculate distances between hands and tools
        print("\nDistance analysis:")

        for hand in hands[:2]:  # Top 2 hands
            hand_pos = np.array(hand['center'])

            print(f"\n{hand['label']} at {hand['center']}:")

            for tool in tools:
                tool_pos = np.array(tool['center'])
                distance_px = np.linalg.norm(hand_pos - tool_pos)

                print(f"  → {tool['label']}: {distance_px:.0f} px away")

                if distance_px < 200:
                    print(f"     ✓ Within reach!")
    else:
        print("\nInsufficient detections for analysis")

    return scene


# Main test runner
if __name__ == "__main__":
    print("="*60)
    print("Multi-Object Detection Examples")
    print("="*60)

    # Run examples
    example_1_detect_multiple_tools()
    example_2_separate_detection()
    example_3_detect_hands_and_tools()
    example_4_filter_by_geometry()
    example_5_3d_positions()
    example_6_visualize_categories()
    example_7_track_specific_hand()
    example_8_workspace_analysis()

    print("\n" + "="*60)
    print("✓ All examples complete!")
    print("="*60)
