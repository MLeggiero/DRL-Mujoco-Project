#!/usr/bin/env python3
"""
Simple PPO-style training for G1 Reach-Touch task
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from g1_rl_environment import G1ReachTouchEnv
except ImportError as e:
    print(f"Error importing G1ReachTouchEnv: {e}")
    print("Make sure g1_rl_environment.py is in the same directory")
    sys.exit(1)

class SimplePolicy:
    """Simple neural network policy (simplified for demonstration)"""
    
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Simple linear policy with one hidden layer
        self.W1 = np.random.randn(64, obs_dim) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(action_dim, 64) * 0.1
        self.b2 = np.zeros(action_dim)
        
        self.learning_rate = 0.001
        
    def forward(self, obs):
        """Forward pass through policy"""
        # Convert observation to vector
        if isinstance(obs, dict):
            obs_vec = self._dict_to_vector(obs)
        else:
            obs_vec = obs
            
        # Ensure obs_vec is the right size
        if len(obs_vec) != self.obs_dim:
            # Pad or truncate to match expected size
            if len(obs_vec) < self.obs_dim:
                obs_vec = np.pad(obs_vec, (0, self.obs_dim - len(obs_vec)), 'constant')
            else:
                obs_vec = obs_vec[:self.obs_dim]
        
        # Simple 2-layer network
        h1 = np.tanh(np.dot(self.W1, obs_vec) + self.b1)
        action = np.tanh(np.dot(self.W2, h1) + self.b2)
        
        return action
    
    def _dict_to_vector(self, obs_dict):
        """Convert observation dictionary to vector"""
        obs_parts = []
        
        # End effector position (3D)
        if 'end_effector_pos' in obs_dict:
            obs_parts.append(obs_dict['end_effector_pos'])
        
        # Target position (3D)
        if 'target_position' in obs_dict:
            obs_parts.append(obs_dict['target_position'])
        
        # Distance to target (1D)
        if 'distance_to_target' in obs_dict:
            obs_parts.append([obs_dict['distance_to_target']])
        
        # Joint positions (limited to first 10 to keep obs size manageable)
        if 'robot_qpos' in obs_dict:
            qpos = obs_dict['robot_qpos'][:10]  # First 10 joints
            obs_parts.append(qpos)
        
        # Joint velocities (limited to first 10)
        if 'robot_qvel' in obs_dict:
            qvel = obs_dict['robot_qvel'][:10]  # First 10 joint velocities
            obs_parts.append(qvel)
            
        if obs_parts:
            return np.concatenate(obs_parts)
        else:
            return np.zeros(self.obs_dim)
    
    def update(self, obs_batch, action_batch, reward_batch):
        """Simple policy gradient update"""
        # Very simplified update - real PPO is much more complex
        batch_size = len(obs_batch)
        if batch_size == 0:
            return
        
        total_grad_w1 = np.zeros_like(self.W1)
        total_grad_b1 = np.zeros_like(self.b1)
        total_grad_w2 = np.zeros_like(self.W2)
        total_grad_b2 = np.zeros_like(self.b2)
        
        for i in range(batch_size):
            obs_vec = self._dict_to_vector(obs_batch[i]) if isinstance(obs_batch[i], dict) else obs_batch[i]
            action = action_batch[i]
            reward = reward_batch[i]
            
            # Ensure proper sizing
            if len(obs_vec) != self.obs_dim:
                if len(obs_vec) < self.obs_dim:
                    obs_vec = np.pad(obs_vec, (0, self.obs_dim - len(obs_vec)), 'constant')
                else:
                    obs_vec = obs_vec[:self.obs_dim]
            
            # Forward pass
            h1 = np.tanh(np.dot(self.W1, obs_vec) + self.b1)
            pred_action = np.tanh(np.dot(self.W2, h1) + self.b2)
            
            # Simple gradient computation (very simplified!)
            action_error = action - pred_action
            
            # Backprop (simplified)
            grad_w2 = np.outer(action_error * reward, h1)
            grad_b2 = action_error * reward
            
            grad_h1 = np.dot(self.W2.T, action_error * reward)
            grad_tanh = 1 - h1**2
            
            grad_w1 = np.outer(grad_h1 * grad_tanh, obs_vec)
            grad_b1 = grad_h1 * grad_tanh
            
            total_grad_w1 += grad_w1
            total_grad_b1 += grad_b1
            total_grad_w2 += grad_w2
            total_grad_b2 += grad_b2
        
        # Update parameters
        lr = self.learning_rate / batch_size
        self.W1 += lr * total_grad_w1
        self.b1 += lr * total_grad_b1
        self.W2 += lr * total_grad_w2
        self.b2 += lr * total_grad_b2

class G1Trainer:
    """G1 RL Trainer"""
    
    def __init__(self):
        try:
            self.env = G1ReachTouchEnv()
        except Exception as e:
            print(f"Failed to create environment: {e}")
            raise
        
        # Policy dimensions (estimated based on observation space)
        self.obs_dim = 27  # 3+3+1+10+10 (end_eff_pos + target_pos + distance + qpos + qvel)
        self.action_dim = max(1, self.env.n_actions)
        
        print(f"Creating policy with obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        print(f"Controlling: Right arm + torso only (legs locked for stability)")
        self.policy = SimplePolicy(self.obs_dim, self.action_dim)
        
        # Training parameters
        self.batch_size = 32
        self.max_episodes = 1000
        self.max_steps_per_episode = 200
        
        # Logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        
    def collect_episode(self, render=False):
        """Collect one episode of experience"""
        obs = self.env.reset()
        
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        
        total_reward = 0
        
        for step in range(self.max_steps_per_episode):
            # Get action from policy
            try:
                action = self.policy.forward(obs)
            except Exception as e:
                print(f"Error in policy forward: {e}")
                action = np.random.uniform(-1, 1, self.action_dim)
            
            # Add exploration noise
            noise = np.random.normal(0, 0.1, action.shape)
            action_with_noise = action + noise
            action_with_noise = np.clip(action_with_noise, -1, 1)
            
            # Step environment
            try:
                next_obs, reward, done, info = self.env.step(action_with_noise)
            except Exception as e:
                print(f"Error in environment step: {e}")
                break
            
            # Store experience
            episode_obs.append(obs)
            episode_actions.append(action_with_noise)
            episode_rewards.append(reward)
            
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        success = info.get('success', False) if 'info' in locals() else False
        return episode_obs, episode_actions, episode_rewards, total_reward, success
    
    def train(self, render_every=50):
        """Main training loop"""
        print("Starting G1 Reach-Touch Training")
        print("=" * 60)
        
        start_time = time.time()
        recent_successes = []
        
        for episode in range(self.max_episodes):
            # Collect episode
            render_this_episode = (episode % render_every == 0) and episode > 0
            
            obs_batch, action_batch, reward_batch, total_reward, success = self.collect_episode(
                render=render_this_episode
            )
            
            # Simple reward processing (discounted returns)
            processed_rewards = []
            cumulative = 0
            discount = 0.99
            for r in reversed(reward_batch):
                cumulative = r + discount * cumulative
                processed_rewards.append(cumulative)
            processed_rewards.reverse()
            
            # Update policy
            if len(obs_batch) > 0:
                try:
                    self.policy.update(obs_batch, action_batch, processed_rewards)
                except Exception as e:
                    print(f"Error in policy update: {e}")
            
            # Logging
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(len(obs_batch))
            recent_successes.append(success)
            
            # Calculate recent success rate
            if len(recent_successes) > 50:
                recent_successes.pop(0)
            success_rate = np.mean(recent_successes) * 100
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
                
                print(f"Episode {episode:4d}: "
                      f"Reward={avg_reward:6.1f}, "
                      f"Length={avg_length:5.1f}, "
                      f"Success={success_rate:5.1f}%")
            
            # Early stopping if doing well
            if len(recent_successes) >= 20 and np.mean(recent_successes[-20:]) > 0.8:
                print(f"Training completed! High success rate achieved.")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining finished in {training_time:.1f} seconds")
        
        # Save results
        self.save_results()
        self.plot_results()
    
    def save_results(self):
        """Save training results and policy"""
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_weights': {
                'W1': self.policy.W1,
                'b1': self.policy.b1,
                'W2': self.policy.W2,
                'b2': self.policy.b2
            }
        }
        
        try:
            with open('g1_training_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            print("Training results saved to 'g1_training_results.pkl'")
        except Exception as e:
            print(f"Failed to save results: {e}")
    
    def plot_results(self):
        """Plot training progress"""
        if len(self.episode_rewards) == 0:
            print("No data to plot")
            return
            
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot rewards
            episodes = range(len(self.episode_rewards))
            ax1.plot(episodes, self.episode_rewards, alpha=0.3, label='Episode Reward')
            
            # Moving average
            if len(self.episode_rewards) > 10:
                moving_avg = []
                for i in range(len(self.episode_rewards)):
                    start_idx = max(0, i - 10)
                    moving_avg.append(np.mean(self.episode_rewards[start_idx:i+1]))
                ax1.plot(episodes, moving_avg, label='Moving Average', linewidth=2)
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('Training Progress - Rewards')
            ax1.legend()
            ax1.grid(True)
            
            # Plot episode lengths
            ax2.plot(episodes, self.episode_lengths, alpha=0.3, label='Episode Length')
            
            # Moving average for lengths
            if len(self.episode_lengths) > 10:
                moving_avg_len = []
                for i in range(len(self.episode_lengths)):
                    start_idx = max(0, i - 10)
                    moving_avg_len.append(np.mean(self.episode_lengths[start_idx:i+1]))
                ax2.plot(episodes, moving_avg_len, label='Moving Average', linewidth=2)
            
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode Length')
            ax2.set_title('Training Progress - Episode Lengths')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('g1_training_progress.png')
            plt.show()
            
            print("Training plots saved to 'g1_training_progress.png'")
            
        except Exception as e:
            print(f"Failed to create plots: {e}")
    
    def test_trained_policy(self, num_episodes=5):
        """Test the trained policy"""
        print(f"\nTesting trained policy for {num_episodes} episodes...")
        
        successes = 0
        total_rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            total_reward = 0
            
            print(f"Test episode {episode + 1}: Target = {self.env.current_target}")
            
            for step in range(self.max_steps_per_episode):
                action = self.policy.forward(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                
                if step % 20 == 0:
                    distance = obs.get('distance_to_target', 0)
                    print(f"  Step {step}: distance = {distance:.3f}")
                
                if done:
                    if info.get('success', False):
                        successes += 1
                        print(f"  SUCCESS! Total reward: {total_reward:.1f}")
                    else:
                        print(f"  Failed. Total reward: {total_reward:.1f}")
                    break
            
            total_rewards.append(total_reward)
        
        success_rate = successes / num_episodes * 100
        avg_reward = np.mean(total_rewards)
        
        print(f"\nTest Results:")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average reward: {avg_reward:.1f}")
        
        self.env.close()

def main():
    """Main training function"""
    
    print("G1 Reach-Touch RL Training")
    print("=" * 60)
    print("This will train the G1 to reach and touch objects on the table.")
    print("Training may take 10-30 minutes depending on your hardware.")
    print()
    
    # Check file structure first
    scene_path = "../unitree_g1/g1_table_box_scene.xml"
    if not os.path.exists(scene_path):
        print("ERROR: Required files not found!")
        print(f"Looking for: {scene_path}")
        print(f"Current directory: {os.getcwd()}")
        print()
        print("Expected file structure:")
        print("  unitree_g1/")
        print("    ├── g1.xml")
        print("    ├── g1_table_box_scene.xml")
        print("    └── assets/")
        print("  ├── g1_rl_environment.py")
        print("  └── g1_training_script.py")
        return
    
    choice = input("Choose training mode:\n"
                  "1. Full training (1000 episodes)\n"
                  "2. Quick test (100 episodes)\n"
                  "3. Test environment only\n"
                  "Enter choice (1-3): ").strip()
    
    try:
        if choice == "1":
            print("Starting full training...")
            trainer = G1Trainer()
            trainer.max_episodes = 1000
            trainer.train(render_every=100)  # Render less often for speed
            trainer.test_trained_policy()
            
        elif choice == "2":
            print("Starting quick test training...")
            trainer = G1Trainer()
            trainer.max_episodes = 100
            trainer.train(render_every=20)
            trainer.test_trained_policy()
            
        elif choice == "3":
            print("Testing environment setup...")
            from g1_rl_environment import test_environment
            success = test_environment()
            if success:
                print("Environment test passed! You can now run training.")
            else:
                print("Environment test failed. Please check the setup.")
            
        else:
            print("Invalid choice!")
            return
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if 'trainer' in locals():
            print("Saving partial results...")
            trainer.save_results()
            trainer.plot_results()
    
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()