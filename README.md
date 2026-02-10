# Unitree G1 Vision-Guided Grasping with RL

A reinforcement learning framework for training the Unitree G1 humanoid robot to perform vision-guided manipulation tasks in MuJoCo simulation. The project integrates RGB-D perception, grasp detection (GraspNet / AnyGrasp), object detection (YOLO / Grounding DINO), and PPO-based RL training via Stable-Baselines3.

## Overview

The robot learns to locate, reach, and grasp objects using a combination of:
- **Proprioceptive RL**: Joint-space control trained with PPO
- **Vision-guided grasping**: RGB-D observations from simulated wrist and head cameras
- **Grasp detection**: GraspNet-1Billion or AnyGrasp for 6-DOF grasp candidates
- **Object detection**: YOLO v8 and Grounding DINO for zero-shot object localization
- **Point cloud processing**: Real-time 3D scene understanding from depth data

## Repository Structure

```
DRL-Mujoco-Project/
├── custom_mujoco_scene/
│   ├── unitree_g1/
│   │   ├── g1.xml                      # Base G1 robot model (MuJoCo Menagerie)
│   │   ├── g1_with_hands.xml           # G1 with dexterous hands
│   │   ├── g1_table_box_scene.xml      # Custom scene: G1 + table + objects
│   │   └── assets/                     # Robot STL meshes
│   ├── g1_scene_launcher.py            # Interactive G1 scene viewer
│   └── playground_test.py              # Quick simulation tests
│
└── tool_use_env/
    ├── hammer_grasp_scene.xml          # MuJoCo scene: robot + table + hammer
    ├── hammer_grasp_rgbd_scene.xml     # Scene with RGBD cameras
    ├── hammer_grasp_environment.py     # Core MuJoCo environment
    ├── hammer_gym_wrapper.py           # Gymnasium wrapper (proprioceptive)
    ├── hammer_rgbd_gym_wrapper.py      # Gymnasium wrapper (RGBD observations)
    ├── vision_guided_grasp_env.py      # Full vision + grasping RL environment
    ├── hybrid_vision_env.py            # Hybrid obs: proprioception + vision
    ├── grasp_detector.py               # Multi-backend grasp detection wrapper
    ├── motion_planner.py               # IK-based trajectory and grasp planning
    ├── generate_pointcloud.py          # RGB-D to point cloud conversion
    ├── segment_hammer.py               # Color/geometry-based segmentation
    ├── camera_utils.py                 # Camera intrinsics and projection utils
    ├── grasp_detector.py               # AnyGrasp / GraspNet / heuristic wrapper
    ├── yolo_detector.py                # YOLOv8 object detection
    ├── grounding_dino_detector.py      # Grounding DINO zero-shot detection
    ├── multi_object_detector.py        # Multi-object detection and tracking
    ├── train_rgbd_sb3.py               # PPO training with RGBD observations
    ├── train_vision_grasp.py           # Vision-guided grasp training
    ├── train_grounding_dino_grasp.py   # Grounding DINO + grasp training
    ├── train_pointcloud_grasp.py       # Point cloud + grasp training
    ├── requirements.txt                # Python dependencies
    └── assets/                         # Robot meshes, hammer OBJ, textures
```

## Features

- **Multi-backend grasp detection**: Supports AnyGrasp (if licensed), GraspNet-1Billion, or heuristic fallback via a unified `GraspDetector` API
- **Zero-shot object detection**: Grounding DINO allows detecting arbitrary objects by text prompt without retraining
- **RGBD observations**: Simulated wrist and head cameras provide depth-aware observations for the RL policy
- **Dexterous hands**: Full 31+ DOF control including finger articulation
- **Stable-Baselines3 compatible**: All environments expose a standard Gymnasium interface

## Prerequisites

- Python 3.10+
- MuJoCo 3.3.6+
- CUDA-capable GPU (for grasp detection inference)
- Ubuntu 20.04+ (or WSL2 on Windows)

## Installation

### 1. Create environment

```bash
python3 -m venv mujoco_env
source mujoco_env/bin/activate
pip install --upgrade pip
```

### 2. Install dependencies

```bash
cd tool_use_env
pip install -r requirements.txt
```

Core dependencies:
- `mujoco>=3.3.6`
- `stable-baselines3[extra]`
- `gymnasium`
- `torch` with CUDA
- `open3d`
- `ultralytics` (YOLOv8)

### 3. (Optional) GraspNet-1Billion

For GPU grasp detection, install [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) and set the `GRASPNET_PATH` environment variable:

```bash
export GRASPNET_PATH=/path/to/graspnet-baseline
```

### 4. (Optional) AnyGrasp

AnyGrasp requires a separate license from the authors. If installed, set:

```bash
export ANYGRASP_PATH=/path/to/anygrasp_sdk
```

The `grasp_detector.py` wrapper will automatically prefer AnyGrasp → GraspNet → heuristic based on availability.

### 5. (Optional) Grounding DINO

```bash
cd tool_use_env
bash install_grounding_dino.sh
```

## Quick Start

### View the G1 robot scene

```bash
source mujoco_env/bin/activate
python custom_mujoco_scene/g1_scene_launcher.py
```

### Test the hammer grasping environment

```bash
cd tool_use_env
python visualize_hammer_grasp.py
```

### Train with PPO (proprioceptive)

```bash
cd tool_use_env
python train_rgbd_sb3.py
```

### Train with vision-guided grasping

```bash
cd tool_use_env
python train_vision_grasp.py
```

### Train with Grounding DINO object detection

```bash
cd tool_use_env
python train_grounding_dino_grasp.py
```

## Environment Details

### Observation Space

Environments offer three observation modes:

| Mode | Dimensions | Contents |
|------|-----------|---------|
| Proprioceptive | ~40 | Joint positions/velocities, hand/object positions |
| RGBD | Variable | RGB image + depth map from wrist/head cameras |
| Hybrid | Variable | Proprioception + compressed vision features |

### Action Space

- Joint torque control for arms and hands (~17-31 DOF depending on config)
- Range: [-1, 1] normalized, scaled internally

### Reward Function

```
reward = distance_reward + contact_reward + grasp_reward + smoothness_penalty
```

- `distance_reward`: -0.5 * hand-to-object distance
- `contact_reward`: +0.5 per step when hand contacts object
- `grasp_reward`: +1.0 for sustained stable grasp (>10 contact frames)
- `smoothness_penalty`: -0.01 * action jerk

## Grasp Detection

The `GraspDetector` class in `grasp_detector.py` provides a unified interface:

```python
from grasp_detector import GraspDetector

detector = GraspDetector(backend="auto")  # auto-selects best available
grasps = detector.detect(pointcloud_xyz, pointcloud_rgb)

best = grasps[0]
print(f"Position: {best.translation}")
print(f"Rotation: {best.rotation_matrix}")
print(f"Width: {best.width:.3f}m, Score: {best.score:.3f}")
```

Backends: `"anygrasp"`, `"graspnet"`, `"heuristic"`, `"auto"`

## Object Detection

```python
from yolo_detector import YOLODetector
from grounding_dino_detector import GroundingDINODetector

# YOLO (trained classes)
yolo = YOLODetector()
detections = yolo.detect(rgb_image)

# Grounding DINO (zero-shot, text-prompted)
dino = GroundingDINODetector()
detections = dino.detect(rgb_image, text_prompt="hammer on a table")
```

## MuJoCo Scene Customization

### Modify the hammer scene

Edit `tool_use_env/hammer_grasp_scene.xml` to change object properties:

```xml
<!-- Change hammer head mass -->
<inertial pos="0.0 0.0 -0.04" mass="1.2" diaginertia="0.001 0.001 0.001"/>

<!-- Change hammer friction -->
<geom name="hammer_head" friction="0.7" density="900"/>
```

### Add new objects

1. Add OBJ/STL file to `tool_use_env/assets/`
2. Reference it in the XML scene file
3. Update the observation/reward logic in the environment

## Acknowledgments

- **[MuJoCo](https://mujoco.org/)** - Physics simulation by DeepMind
- **[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)** - G1 robot model
- **[Unitree Robotics](https://www.unitree.com/g1)** - G1 humanoid hardware
- **[GraspNet-1Billion](https://graspnet.net/)** - Grasp detection model (Fang et al., CVPR 2020)
- **[AnyGrasp](https://graspnet.net/anygrasp.html)** - General grasp detection SDK
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** - RL algorithms
- **[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)** - Zero-shot object detection

## License

Custom RL environments and training scripts are released under MIT.
The Unitree G1 robot model is from MuJoCo Menagerie (BSD-3-Clause, Unitree Robotics).
AnyGrasp and GraspNet are subject to their own licenses — see their respective repositories.

## Citation

If you use this work in your research:

```bibtex
@misc{unitree_g1_vision_grasp,
  title={Unitree G1 Vision-Guided Grasping with Reinforcement Learning},
  author={Leggiero, M.},
  year={2025},
  url={https://github.com/mleggiero/DRL-Mujoco-Project}
}
```
