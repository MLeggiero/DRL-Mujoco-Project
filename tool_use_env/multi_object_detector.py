#!/usr/bin/env python3
"""
Enhanced Multi-Object Grounding DINO Detector.

Supports:
- Multiple object types in one detection
- Separate detection categories (tools, hands, objects)
- Batch detection for efficiency
- Hand/gripper tracking
"""

import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Union


class MultiObjectDetector:
    """Enhanced Grounding DINO detector for multiple objects and categories."""

    # Predefined object categories
    TOOL_PROMPTS = [
        "hammer",
        "screwdriver",
        "wrench",
        "pliers",
        "drill",
        "saw",
        "chisel",
        "tape measure"
    ]

    HAND_PROMPTS = [
        "robot hand",
        "robot gripper",
        "robotic gripper",
        "mechanical hand",
        "robot arm end effector"
    ]

    OBJECT_PROMPTS = [
        "nail",
        "screw",
        "bolt",
        "nut",
        "wood block",
        "metal piece"
    ]

    def __init__(self, verbose=True):
        """Initialize multi-object detector."""
        from groundingdino.util.inference import load_model
        from huggingface_hub import hf_hub_download

        self.verbose = verbose

        if self.verbose:
            print("Loading Grounding DINO model for multi-object detection...")

        # Download model files
        cache_dir = Path.home() / ".cache" / "groundingdino"
        cache_dir.mkdir(parents=True, exist_ok=True)

        config_path = hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO",
            filename="GroundingDINO_SwinT_OGC.cfg.py",
            cache_dir=cache_dir
        )

        checkpoint_path = hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO",
            filename="groundingdino_swint_ogc.pth",
            cache_dir=cache_dir
        )

        # Load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(config_path, checkpoint_path, device=self.device)

        if self.verbose:
            print(f"✓ Multi-Object Detector initialized on {self.device}")

    def detect_multiple_objects(
        self,
        rgb_image: np.ndarray,
        prompts: List[str],
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        combine_results: bool = True
    ) -> Union[List[Dict], Dict[str, List[Dict]]]:
        """
        Detect multiple object types in one image.

        Args:
            rgb_image: RGB image (H, W, 3)
            prompts: List of text prompts to detect
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            combine_results: If True, return all detections in one list
                           If False, return dict mapping prompt -> detections

        Returns:
            If combine_results=True:
                List of detections with 'prompt' field indicating which prompt matched
            If combine_results=False:
                Dict: {prompt: [detections]}
        """
        if combine_results:
            # Single detection call with all prompts combined
            combined_prompt = " . ".join(prompts)
            detections = self.detect(rgb_image, combined_prompt, box_threshold, text_threshold)

            # Add prompt field based on label
            for det in detections:
                # Match detection label to original prompt
                label = det['label'].lower()
                det['prompt'] = self._match_label_to_prompt(label, prompts)

            return detections
        else:
            # Separate detection for each prompt
            results = {}
            for prompt in prompts:
                detections = self.detect(rgb_image, prompt, box_threshold, text_threshold)
                results[prompt] = detections
            return results

    def detect_tools(
        self,
        rgb_image: np.ndarray,
        specific_tools: Optional[List[str]] = None,
        box_threshold: float = 0.30,
        apply_geometry_filter: bool = True
    ) -> List[Dict]:
        """
        Detect tools in the image.

        Args:
            rgb_image: RGB image
            specific_tools: Specific tool names, or None for all common tools
            box_threshold: Detection threshold
            apply_geometry_filter: Apply geometry-based filtering

        Returns:
            List of tool detections
        """
        if specific_tools is None:
            prompts = self.TOOL_PROMPTS
        else:
            prompts = specific_tools

        detections = self.detect_multiple_objects(
            rgb_image,
            prompts,
            box_threshold=box_threshold,
            combine_results=True
        )

        # Apply geometry filtering to remove false positives
        if apply_geometry_filter:
            detections = self.filter_by_category(detections, 'tool', rgb_image.shape)

        return detections

    def detect_hands(
        self,
        rgb_image: np.ndarray,
        box_threshold: float = 0.25,
        apply_geometry_filter: bool = True
    ) -> List[Dict]:
        """
        Detect robot hands/grippers in the image.

        Args:
            rgb_image: RGB image
            box_threshold: Detection threshold (lower for hands)
            apply_geometry_filter: Apply geometry-based filtering

        Returns:
            List of hand/gripper detections
        """
        detections = self.detect_multiple_objects(
            rgb_image,
            self.HAND_PROMPTS,
            box_threshold=box_threshold,
            combine_results=True
        )

        # Apply geometry filtering to remove false positives
        if apply_geometry_filter:
            detections = self.filter_by_category(detections, 'hand', rgb_image.shape)

        return detections

    def detect_scene(
        self,
        rgb_image: np.ndarray,
        include_tools: bool = True,
        include_hands: bool = True,
        include_objects: bool = True,
        tool_threshold: float = 0.30,
        hand_threshold: float = 0.25,
        object_threshold: float = 0.30,
        remove_overlaps: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Detect complete scene with different object categories.

        Args:
            rgb_image: RGB image
            include_tools: Detect tools
            include_hands: Detect robot hands
            include_objects: Detect other objects
            tool_threshold: Tool detection threshold
            hand_threshold: Hand detection threshold
            object_threshold: Object detection threshold
            remove_overlaps: Remove overlapping detections between categories

        Returns:
            Dict with keys: 'tools', 'hands', 'objects'
            Each containing list of detections
        """
        results = {
            'tools': [],
            'hands': [],
            'objects': []
        }

        if include_tools:
            results['tools'] = self.detect_tools(rgb_image, box_threshold=tool_threshold)

        if include_hands:
            results['hands'] = self.detect_hands(rgb_image, box_threshold=hand_threshold)

        if include_objects:
            results['objects'] = self.detect_multiple_objects(
                rgb_image,
                self.OBJECT_PROMPTS,
                box_threshold=object_threshold,
                combine_results=True
            )

        # Remove overlapping detections (same object detected in multiple categories)
        if remove_overlaps:
            results = self._remove_overlapping_detections(results)

        return results

    def detect(
        self,
        rgb_image: np.ndarray,
        text_prompt: str = "hammer",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> List[Dict]:
        """
        Detect objects using text prompt (single or combined).

        Args:
            rgb_image: RGB image (H, W, 3) uint8
            text_prompt: Text prompt(s). Use " . " to separate multiple objects
                        Example: "hammer . screwdriver . wrench"
            box_threshold: Confidence threshold for detection
            text_threshold: Text matching threshold

        Returns:
            List of detections with boxes, scores, labels, and geometry
        """
        from groundingdino.util.inference import predict, load_image
        import tempfile
        import os

        # Convert to PIL Image
        if isinstance(rgb_image, np.ndarray):
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = rgb_image

        # Save to temp file and load with groundingdino's loader
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            pil_image.save(temp_path)

        try:
            # Load image using groundingdino's helper
            image_source, image_tensor = load_image(temp_path)

            # Run detection
            boxes, confidences, labels = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
        finally:
            os.unlink(temp_path)

        # Convert to list of dicts
        results = []
        h, w = rgb_image.shape[:2]

        for i in range(len(boxes)):
            box = boxes[i]  # normalized coords [cx, cy, w, h]

            # Convert from center format to corner format and scale to pixels
            cx, cy, bw, bh = box
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            # Clamp to image bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))

            # Calculate geometry
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(height, 1)
            area = width * height

            results.append({
                'bbox': [x1, y1, x2, y2],
                'bbox_normalized': [x1/w, y1/h, x2/w, y2/h],
                'confidence': float(confidences[i]),
                'label': str(labels[i]) if labels[i] else text_prompt.split('.')[0].strip(),
                'center': [(x1+x2)//2, (y1+y2)//2],
                'geometry': {
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'orientation': 'horizontal' if aspect_ratio > 1.5 else ('vertical' if aspect_ratio < 0.67 else 'square')
                }
            })

        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results

    def _remove_overlapping_detections(
        self,
        scene_detections: Dict[str, List[Dict]],
        iou_threshold: float = 0.3
    ) -> Dict[str, List[Dict]]:
        """
        Remove overlapping detections between categories.

        If the same object is detected in multiple categories,
        keep it only in the most appropriate category based on geometry.

        Args:
            scene_detections: Dict with 'tools', 'hands', 'objects'
            iou_threshold: IoU threshold for considering detections as overlapping

        Returns:
            Cleaned scene detections
        """
        def compute_iou(box1, box2):
            """Compute IoU between two bounding boxes."""
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2

            # Intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)

            if x2_i < x1_i or y2_i < y1_i:
                return 0.0

            intersection = (x2_i - x1_i) * (y2_i - y1_i)

            # Union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        cleaned = {
            'tools': [],
            'hands': [],
            'objects': []
        }

        # Collect all detections with their categories
        all_dets = []
        for category, detections in scene_detections.items():
            for det in detections:
                all_dets.append((category, det))

        # Track which detections to keep
        to_remove = set()

        # Check for overlaps between categories
        for i, (cat1, det1) in enumerate(all_dets):
            if i in to_remove:
                continue

            for j, (cat2, det2) in enumerate(all_dets[i+1:], start=i+1):
                if j in to_remove or cat1 == cat2:
                    continue

                iou = compute_iou(det1['bbox'], det2['bbox'])

                if iou > iou_threshold:
                    # Overlapping detections from different categories
                    # Decide which one to keep based on geometry

                    # Priority: Keep the one that better matches its category
                    geom1 = det1['geometry']
                    geom2 = det2['geometry']

                    # Score how well each matches its category
                    def category_score(det, cat):
                        geom = det['geometry']
                        if cat == 'tool':
                            # Tools should be horizontal and small
                            return geom['aspect_ratio'] * (1.0 / (geom['area'] + 1))
                        elif cat == 'hand':
                            # Hands should be large and square/vertical
                            return geom['area'] * (1.0 / (geom['aspect_ratio'] + 0.1))
                        else:
                            return 1.0

                    score1 = category_score(det1, cat1)
                    score2 = category_score(det2, cat2)

                    # Remove the one with lower score
                    if score1 < score2:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

        # Add non-removed detections to cleaned results
        for i, (category, det) in enumerate(all_dets):
            if i not in to_remove:
                cleaned[category].append(det)

        return cleaned

    def filter_by_category(
        self,
        detections: List[Dict],
        category: str,
        image_shape: tuple
    ) -> List[Dict]:
        """
        Filter detections by category using geometry heuristics.

        Args:
            detections: List of detections
            category: 'tool', 'hand', or 'object'
            image_shape: (height, width) of image

        Returns:
            Filtered detections
        """
        filtered = []
        h, w = image_shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            geom = det['geometry']

            if category == 'tool':
                # Tools are typically:
                # - Horizontal (aspect > 2.0) - much more elongated than hands
                # - Upper part of image (y1 < 40% of height) - far from camera
                # - Small to medium size (not huge like robot arms)
                # - NOT in the lower 60% where hands typically are
                is_tool = (
                    geom['aspect_ratio'] > 2.0 and  # More strict: must be horizontal
                    y1 < h * 0.4 and                # Must be in upper part
                    y2 < h * 0.6 and                # Must end before lower part
                    geom['width'] < w * 0.3 and     # Not too wide
                    geom['area'] < (w * h * 0.08)   # Not too large
                )
                if is_tool:
                    filtered.append(det)

            elif category == 'hand':
                # Robot hands are typically:
                # - Vertical or square (aspect < 1.2)
                # - Lower part of image (y2 > 50% of height) - close to camera
                # - Large (area > 5% of image)
                # - Tall (height > 25% of image height)
                is_hand = (
                    geom['aspect_ratio'] < 1.2 and      # Vertical or square
                    y2 > h * 0.5 and                     # Must extend into lower half
                    geom['area'] > (w * h * 0.05) and    # Must be large
                    geom['height'] > h * 0.25            # Must be tall
                )
                if is_hand:
                    filtered.append(det)

            elif category == 'object':
                # Small objects (nails, screws, etc.)
                # - Small
                # - Any position
                is_object = geom['area'] < (w * h * 0.1)
                if is_object:
                    filtered.append(det)

        return filtered

    def get_3d_position(
        self,
        detection: Dict,
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Convert 2D detection to 3D position using depth.

        Args:
            detection: Detection dict from detect()
            depth_image: Depth map (H, W) in meters
            camera_intrinsics: 3x3 camera matrix K

        Returns:
            3D position [x, y, z] in camera frame
        """
        # Get center of bounding box
        cx, cy = detection['center']

        # Get depth at center
        depth_value = depth_image[cy, cx]

        # Back-project to 3D
        K = camera_intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx_cam, cy_cam = K[0, 2], K[1, 2]

        z = depth_value
        x = (cx - cx_cam) * z / fx
        y = (cy - cy_cam) * z / fy

        return np.array([x, y, z])

    def visualize_detections(
        self,
        rgb_image: np.ndarray,
        detections: Union[List[Dict], Dict[str, List[Dict]]],
        save_path: Optional[str] = None,
        show_geometry: bool = True
    ) -> np.ndarray:
        """
        Visualize detections on image with color-coded categories.

        Args:
            rgb_image: Input RGB image
            detections: List of detections OR dict of {category: detections}
            save_path: Optional path to save visualization
            show_geometry: Show aspect ratio and size info

        Returns:
            Annotated image
        """
        img = rgb_image.copy()

        # Color scheme
        colors = {
            'tools': (0, 255, 0),      # Green
            'hands': (255, 0, 0),      # Red
            'objects': (0, 0, 255),    # Blue
            'default': (0, 255, 255)   # Cyan
        }

        # Handle both formats
        if isinstance(detections, dict):
            # Dict format: {category: [detections]}
            for category, det_list in detections.items():
                color = colors.get(category, colors['default'])
                for det in det_list:
                    self._draw_detection(img, det, color, category, show_geometry)
        else:
            # List format
            color = colors['default']
            for det in detections:
                category = det.get('prompt', det.get('label', 'unknown'))
                if 'hand' in category.lower() or 'gripper' in category.lower():
                    color = colors['hands']
                elif any(tool in category.lower() for tool in ['hammer', 'screwdriver', 'wrench', 'tool']):
                    color = colors['tools']
                else:
                    color = colors['objects']

                self._draw_detection(img, det, color, category, show_geometry)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if self.verbose:
                print(f"✓ Visualization saved to {save_path}")

        return img

    def _draw_detection(
        self,
        img: np.ndarray,
        det: Dict,
        color: tuple,
        label: str,
        show_geometry: bool
    ):
        """Draw a single detection on image."""
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Create label text
        if show_geometry and 'geometry' in det:
            geom = det['geometry']
            text = f"{label}: {conf:.1%} (AR:{geom['aspect_ratio']:.2f})"
        else:
            text = f"{label}: {conf:.1%}"

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _match_label_to_prompt(self, label: str, prompts: List[str]) -> str:
        """Match detection label to original prompt."""
        label_lower = label.lower()

        # Try exact match
        for prompt in prompts:
            if prompt.lower() in label_lower or label_lower in prompt.lower():
                return prompt

        # Return first prompt as default
        return prompts[0] if prompts else "unknown"

    def summarize_scene(
        self,
        scene_detections: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Generate summary statistics for detected scene.

        Args:
            scene_detections: Dict from detect_scene()

        Returns:
            Summary dict with counts and statistics
        """
        summary = {
            'counts': {},
            'confidence_stats': {},
            'all_objects': []
        }

        for category, detections in scene_detections.items():
            summary['counts'][category] = len(detections)

            if len(detections) > 0:
                confidences = [d['confidence'] for d in detections]
                summary['confidence_stats'][category] = {
                    'mean': np.mean(confidences),
                    'max': np.max(confidences),
                    'min': np.min(confidences)
                }

                # Collect all detections
                for det in detections:
                    summary['all_objects'].append({
                        'category': category,
                        'label': det['label'],
                        'confidence': det['confidence'],
                        'bbox': det['bbox']
                    })

        summary['total_objects'] = len(summary['all_objects'])

        return summary


# Convenience function
def create_detector(verbose=True) -> MultiObjectDetector:
    """Create a multi-object detector instance."""
    return MultiObjectDetector(verbose=verbose)


# Test function
if __name__ == "__main__":
    print("="*60)
    print("Testing Multi-Object Detector")
    print("="*60)

    # Initialize
    detector = MultiObjectDetector()

    # Load test image
    image_path = "pointcloud_data/rgb.png"
    print(f"\nLoading image: {image_path}")
    rgb = cv2.imread(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    print(f"\n{'='*60}")
    print("Test 1: Detect Multiple Specific Tools")
    print(f"{'='*60}")

    tools = ["hammer", "screwdriver", "wrench"]
    detections = detector.detect_multiple_objects(rgb, tools, box_threshold=0.25)
    print(f"Found {len(detections)} detections")
    for i, det in enumerate(detections[:5]):
        print(f"  {i+1}. {det['label']}: {det['confidence']:.1%} at {det['bbox']}")

    print(f"\n{'='*60}")
    print("Test 2: Detect Complete Scene")
    print(f"{'='*60}")

    scene = detector.detect_scene(
        rgb,
        include_tools=True,
        include_hands=True,
        include_objects=False,
        tool_threshold=0.30,
        hand_threshold=0.25
    )

    print(f"Tools found: {len(scene['tools'])}")
    for det in scene['tools']:
        print(f"  - {det['label']}: {det['confidence']:.1%}")

    print(f"\nHands found: {len(scene['hands'])}")
    for det in scene['hands']:
        print(f"  - {det['label']}: {det['confidence']:.1%}")

    print(f"\n{'='*60}")
    print("Test 3: Scene Summary")
    print(f"{'='*60}")

    summary = detector.summarize_scene(scene)
    print(f"Total objects: {summary['total_objects']}")
    print(f"Counts: {summary['counts']}")
    print(f"Confidence stats: {summary['confidence_stats']}")

    print(f"\n{'='*60}")
    print("Test 4: Visualize All Detections")
    print(f"{'='*60}")

    # Combine all detections for visualization
    vis_img = detector.visualize_detections(
        rgb,
        scene,
        save_path="multi_object_detection.png",
        show_geometry=True
    )

    print("\n✓ Multi-object detection test complete!")
    print(f"Visualization saved to: multi_object_detection.png")
