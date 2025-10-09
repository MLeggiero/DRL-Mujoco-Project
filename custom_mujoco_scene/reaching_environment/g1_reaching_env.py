"""
G1 Reaching Environment for MuJoCo
This environment works with standard MuJoCo (not mujoco_playground)
"""
import numpy as np
import mujoco

class G1ReachingEnv:
    """Custom environment where Unitree G1 reaches for a red box"""
    
    def __init__(self, xml_path='g1_table_box_scene.xml', frame_skip=5, max_episode_steps=1000):
        """
        Args:
            xml_path: Path to the MuJoCo XML file
            frame_skip: Number of simulation steps per action
            max_episode_steps: Maximum steps before episode terminates
        """
        self.xml_path = xml_path
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Cache body/site IDs
        self._setup_ids()
        
        # Episode tracking
        self.current_step = 0
        
    def _setup_ids(self):
        """Cache IDs for faster access"""
        try:
            # Try to find right_hand_site if it exists
            self.right_hand_id = self.model.site('right_hand_site').id
            self.use_site = True
        except:
            # Fall back to right_wrist_yaw_link body
            self.right_hand_id = self.model.body('right_wrist_yaw_link').id
            self.use_site = False
            
        self.red_box_id = self.model.body('red_box').id
        self.pelvis_id = self.model.body('pelvis').id
        
    def reset(self, seed=None):
        """Reset the environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random noise to initial state for diversity
        self.data.qpos[:] += np.random.uniform(-0.01, 0.01, size=self.data.qpos.shape)
        self.data.qvel[:] = np.random.uniform(-0.01, 0.01, size=self.data.qvel.shape)
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        return self._get_obs()
    
    def step(self, action):
        """Take a step in the environment"""
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action
        self.data.ctrl[:] = action
        
        # Step simulation multiple times
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation, reward, done
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        
        self.current_step += 1
        
        # Create info dict
        info = {
            'hand_position': self._get_hand_position().copy(),
            'target_position': self._get_target_position().copy(),
            'distance': self._get_distance(),
        }
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Get current observation"""
        hand_pos = self._get_hand_position()
        target_pos = self._get_target_position()
        
        obs = np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flatten(),
            hand_pos,
            target_pos,
            hand_pos - target_pos,  # Relative position
        ])
        
        return obs.astype(np.float32)
    
    def _get_hand_position(self):
        """Get right hand position"""
        if self.use_site:
            return self.data.site_xpos[self.right_hand_id].copy()
        else:
            return self.data.xpos[self.right_hand_id].copy()
    
    def _get_target_position(self):
        """Get target (red box) position"""
        return self.data.xpos[self.red_box_id].copy()
    
    def _get_distance(self):
        """Get distance from hand to target"""
        return np.linalg.norm(self._get_hand_position() - self._get_target_position())
    
    def _compute_reward(self):
        """Compute reward based on distance to target"""
        distance = self._get_distance()
        
        # Main reward: negative distance (closer = better)
        reward = -distance
        
        # Bonus for getting very close
        if distance < 0.05:
            reward += 10.0
        
        # Small penalty for large actions (encourage smooth motion)
        action_penalty = -0.001 * np.sum(np.square(self.data.ctrl))
        reward += action_penalty
        
        return float(reward)
    
    def _is_done(self):
        """Check if episode should terminate"""
        # Terminate if robot falls
        pelvis_height = self.data.xpos[self.pelvis_id][2]
        if pelvis_height < 0.4:
            return True
        
        # Terminate if hand reaches target
        if self._get_distance() < 0.03:
            return True
        
        # Terminate if max steps reached
        if self.current_step >= self.max_episode_steps:
            return True
        
        return False
    
    @property
    def observation_space_dim(self):
        """Get observation space dimension"""
        return len(self._get_obs())
    
    @property
    def action_space_dim(self):
        """Get action space dimension"""
        return self.model.nu
    
    def render(self):
        """Render is handled externally via mujoco.viewer"""
        pass