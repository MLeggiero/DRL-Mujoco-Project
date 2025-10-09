# Unitree G1 Reaching Environment for MuJoCo

A reinforcement learning environment where a Unitree G1 humanoid robot learns to reach for a red box on a table using **standard MuJoCo** (no special frameworks required).

## ⚠️ Important Note

This environment uses **standard MuJoCo**, NOT `mujoco_playground`. The original error you encountered was due to trying to import from `mujoco_playground.envs`, which doesn't exist in that package structure. This corrected version works with the standard `mujoco` Python package.

## 📁 File Structure

```
.
├── g1.xml                          # G1 robot definition
├── g1_updated.xml                  # G1 with added hand site marker (optional)
├── g1_table_box_scene.xml          # Complete scene with G1, table, and objects
├── g1_reaching_env.py              # Custom environment (pure MuJoCo)
├── g1_reaching_training.py         # Training script with customizable algorithms
├── g1_reaching_visualize.py        # Visualization and rendering tools
├── assets/                         # STL meshes for G1 robot
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install ONLY standard MuJoCo (NOT mujoco_playground)
pip install mujoco numpy

# Optional: for video recording
pip install imageio imageio-ffmpeg
```

**Do NOT install `mujoco_playground` or `playground` for this project** - they are not needed and were causing the import error.

### 2. Verify Installation

```bash
# Test that MuJoCo is installed correctly
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
```

### 3. Run Random Policy (Baseline)

```bash
# Test with random actions for 10 episodes
python g1_reaching_training.py --mode random --episodes 10

# Visualize the random policy (opens interactive viewer)
python g1_reaching_visualize.py --mode random --episodes 3
```

### 4. Train with Evolutionary Strategy

```bash
# Train a simple neural network policy
python g1_reaching_training.py --mode es --episodes 100 --save trained_policy.pkl

# Visualize the trained policy
python g1_reaching_visualize.py --mode trained --policy trained_policy.pkl --episodes 5
```

### 5. Record Video

```bash
# Record video of trained policy
python g1_reaching_visualize.py --mode record --policy trained_policy.pkl --output demo.mp4

# Record video of random policy
python g1_reaching_visualize.py --mode record --output random_demo.mp4
```

## 🔧 What Was Fixed

### Original Error
```python
from mujoco_playground import envs  # ❌ This doesn't exist!
```

### Fixed Version
```python
import mujoco  # ✓ Use standard MuJoCo
import numpy as np
```

The original code had these issues:
1. **Wrong package**: Tried to import from `mujoco_playground.envs` which doesn't have that structure
2. **Double initialization**: Called `super().__init__()` twice with different XML files
3. **Missing base class**: `envs.base.Base` doesn't exist in mujoco_playground
4. **JAX dependency**: Original assumed JAX was required (it's not for this simple case)

The corrected version:
- Uses standard MuJoCo directly
- Implements a simple custom environment class
- Works with NumPy (no JAX required, but you can add it)
- Properly handles the XML file structure
- Includes working training and visualization scripts

## 🎯 Environment Details

### Observation Space
The observation is a flattened vector containing:
- Robot joint positions (`data.qpos`) - 37 values
- Robot joint velocities (`data.qvel`) - 36 values  
- Right hand position (3D) - 3 values
- Target (red box) position (3D) - 3 values
- Relative position (hand - target) - 3 values
- **Total: 82 dimensions**

### Action Space
- 29 continuous actions (one per actuator)
- Action range: [-1, 1] (normalized)
- Actions are clipped to valid range

### Reward Function
```python
reward = -distance(hand, target)  # Main reward
reward += 10.0 if distance < 0.05 else 0.0  # Bonus for reaching
reward += -0.001 * sum(action²)  # Action penalty for smoothness
```

### Termination Conditions
Episode ends when:
- Robot falls (pelvis height < 0.4m)
- Hand reaches target (distance < 0.03m)
- Maximum steps reached (default 500-1000)

## 🛠️ Customization Guide

### 1. Modify the Reward Function

Edit `g1_reaching_env.py`, method `_compute_reward`:

```python
def _compute_reward(self):
    distance = self._get_distance()
    
    # Your custom reward here
    reward = -distance * 2.0  # Increase penalty weight
    
    # Bigger bonus for reaching
    if distance < 0.05:
        reward += 20.0
    
    # Penalize excessive velocity
    velocity_penalty = -0.01 * np.sum(np.square(self.data.qvel))
    reward += velocity_penalty
    
    # Bonus for hand height (encourage lifting)
    hand_height = self._get_hand_position()[2]
    if hand_height > 0.7:
        reward += 1.0
    
    return reward
```

### 2. Implement Your Own RL Algorithm

The training script provides a simple evolutionary strategy. Replace it with your algorithm:

```python
def train_ppo(num_episodes=1000):
    """Your PPO implementation"""
    env = G1ReachingEnv()
    
    # Initialize your PPO agent
    agent = PPOAgent(
        obs_dim=env.observation_space_dim,
        action_dim=env.action_space_dim
    )
    
    for episode in range(num_episodes):
        # Collect trajectory
        trajectory = run_episode(env, agent.policy)
        
        # Update policy and value function
        agent.update(trajectory)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {trajectory['total_reward']:.2f}")
    
    return agent
```

### 3. Add Multiple Targets

Edit `g1_table_box_scene.xml` to add more objects:

```xml
<!-- Add a blue sphere -->
<body name="blue_sphere">
  <joint name="blue_sphere_joint" type="free"/>
  <geom name="blue_sphere_geom" type="sphere" size="0.05" 
        pos="0.5 0.2 0.75" rgba="0.1 0.1 0.9 1" mass="0.15"/>
</body>
```

Then modify the reward function to track multiple objects:

```python
def _compute_reward(self):
    hand_pos = self._get_hand_position()
    
    # Get positions of multiple targets
    red_box_pos = self.data.xpos[self.red_box_id]
    blue_sphere_pos = self.data.xpos[self.model.body('blue_sphere').id]
    
    # Reward for reaching the closest target
    dist_red = np.linalg.norm(hand_pos - red_box_pos)
    dist_blue = np.linalg.norm(hand_pos - blue_sphere_pos)
    
    min_dist = min(dist_red, dist_blue)
    reward = -min_dist
    
    return reward
```

### 4. Use Different Policy Architectures

Replace `SimpleNeuralPolicy` in `g1_reaching_training.py`:

```python
class LSTMPolicy:
    """Recurrent policy for temporal dependencies"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_state = np.zeros(hidden_dim)
        # Initialize LSTM parameters...
    
    def __call__(self, observation):
        # LSTM forward pass
        # Update hidden state
        # Return action
        pass
```

## 📊 Training Tips

### Monitor Progress
```bash
# Train with verbose output
python g1_reaching_training.py --mode es --episodes 200 --learning_rate 0.1

# Watch training in real-time (requires modification to add periodic visualization)
```

### Hyperparameter Tuning

```bash
# Adjust learning rate
python g1_reaching_training.py --mode es --learning_rate 0.05

# Increase population size for better exploration
python g1_reaching_training.py --mode es --population_size 20

# Longer episodes for more learning
python g1_reaching_training.py --mode es --max_steps 1000
```

### Common Issues & Solutions

**Robot falls immediately:**
```python
# Solution 1: Reduce action magnitude
action = policy(obs) * 0.5  # Scale down actions

# Solution 2: Add stability reward
pelvis_upright = self.data.xpos[self.pelvis_id][2]
reward += pelvis_upright  # Encourage staying upright
```

**Hand doesn't move toward target:**
```python
# Solution: Use shaped reward with intermediate goals
distance = self._get_distance()
if distance < 0.5:
    reward += 5.0  # Bonus for getting within 0.5m
if distance < 0.2:
    reward += 10.0  # Larger bonus for getting closer
```

**Training is slow:**
```python
# Solution 1: Reduce frame_skip
env = G1ReachingEnv(frame_skip=2)  # Faster but more computation

# Solution 2: Use parallel environments (requires multiprocessing)
from multiprocessing import Pool

def train_parallel():
    with Pool(4) as pool:
        results = pool.starmap(run_episode, [(env, policy) for _ in range(4)])
```

## 🎨 Visualization Options

### Interactive Viewer Controls
When using `visualize.py`, the MuJoCo viewer has these controls:
- **Mouse drag**: Rotate camera
- **Scroll wheel**: Zoom in/out  
- **Double-click**: Select and follow body
- **Ctrl + Mouse**: Pan camera
- **Backspace**: Reset camera to default

### Slow Motion Mode
```bash
python g1_reaching_visualize.py --mode trained --policy model.pkl --slow
```

### Compare Multiple Policies
```bash
python g1_reaching_visualize.py --mode compare \
    --policies baseline.pkl improved.pkl final.pkl \
    --episodes 5
```

### Custom Camera Angles
Edit `g1_table_box_scene.xml` to add camera definitions:

```xml
<worldbody>
  <!-- Add custom camera -->
  <camera name="side_view" pos="2 0 1" xyaxes="0 -1 0 0 0 1"/>
  <camera name="top_view" pos="0.5 0 2.5" xyaxes="1 0 0 0 1 0"/>
</worldbody>
```

## 🔬 Advanced Features

### Curriculum Learning
```python
class CurriculumEnv(G1ReachingEnv):
    def __init__(self, difficulty=0.0, **kwargs):
        super().__init__(**kwargs)
        self.difficulty = difficulty  # 0.0 to 1.0
    
    def reset(self, seed=None):
        obs = super().reset(seed)
        
        # Move target further away as difficulty increases
        target_distance = 0.3 + (self.difficulty * 0.5)
        self.data.qpos[self.model.jnt_qposadr[
            self.model.joint('red_box_joint').id
        ]] = target_distance
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

# Train with curriculum
for stage in range(5):
    difficulty = stage / 4.0
    env = CurriculumEnv(difficulty=difficulty)
    # Train for some episodes...
```

### Domain Randomization
```python
def randomize_dynamics(env):
    """Randomize physics parameters for robustness"""
    # Randomize masses
    for i in range(env.model.nbody):
        env.model.body_mass[i] *= np.random.uniform(0.8, 1.2)
    
    # Randomize friction
    for i in range(env.model.ngeom):
        env.model.geom_friction[i, 0] *= np.random.uniform(0.5, 1.5)
    
    # Randomize target position
    target_id = env.model.body('red_box').id
    env.data.qpos[env.model.jnt_qposadr[
        env.model.joint('red_box_joint').id
    ]:env.model.jnt_qposadr[
        env.model.joint('red_box_joint').id
    ] + 3] += np.random.uniform(-0.1, 0.1, size=3)
```

### State-Based Curriculum
```python
def adaptive_curriculum(env, policy, success_threshold=0.7):
    """Automatically adjust difficulty based on success rate"""
    window_size = 10
    recent_successes = []
    difficulty = 0.0
    
    while difficulty < 1.0:
        # Run episode
        trajectory = run_episode(env, policy)
        success = trajectory['infos'][-1]['distance'] < 0.05
        recent_successes.append(success)
        
        # Keep only recent history
        if len(recent_successes) > window_size:
            recent_successes.pop(0)
        
        # Adjust difficulty
        success_rate = np.mean(recent_successes)
        if success_rate > success_threshold:
            difficulty = min(1.0, difficulty + 0.1)
            env.difficulty = difficulty
            print(f"Increasing difficulty to {difficulty:.1f}")
```

## 📝 Example: Integrating Popular RL Libraries

### Using Stable-Baselines3
```python
import gymnasium as gym
from stable_baselines3 import PPO

# Wrap your environment in Gymnasium interface
class GymWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = G1ReachingEnv()
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.env.action_space_dim,)
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.env.observation_space_dim,)
        )
    
    def reset(self, seed=None):
        obs = self.env.reset(seed)
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

# Train with PPO
env = GymWrapper()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_g1_reaching")
```

### Using CleanRL
```python
# See CleanRL documentation for PPO/SAC implementations
# that work directly with NumPy-based environments
```

## 🐛 Troubleshooting

### ImportError: cannot import name 'envs' from 'mujoco_playground'

**Solution**: This is the error from your original code. The fixed version doesn't use `mujoco_playground` at all. Just use:
```bash
pip install mujoco numpy
```

### Missing mesh files

**Error**: `RuntimeError: Error: could not open file 'assets/pelvis.STL'`

**Solution**: Ensure you have the `assets/` folder with all STL mesh files in the same directory as your XML files.

### MuJoCo viewer doesn't open

**Error**: `OSError: cannot open shared object file: libGL.so.1`

**Solution** (Linux):
```bash
sudo apt-get install libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6
```

### Robot behavior is unstable

**Solutions**:
1. Reduce frame_skip: `G1ReachingEnv(frame_skip=2)`
2. Scale down actions: `action = policy(obs) * 0.5`
3. Add action smoothing
4. Increase control gains in XML

### Training doesn't improve

**Solutions**:
1. Check reward function - ensure it's properly shaped
2. Increase population size: `--population_size 20`
3. Adjust learning rate: `--learning_rate 0.05`
4. Use longer episodes: `--max_steps 1000`
5. Try a different RL algorithm (PPO, SAC, etc.)

## 📚 Additional Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- [MJCF XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

## 📄 License

This code uses the Unitree G1 model and MuJoCo. Please respect their respective licenses.

---

**Happy Training! 🤖**

If you encounter any issues, double-check that you're NOT trying to install or import `mujoco_playground` - this implementation uses standard MuJoCo only!