"""
PPO Training Script for G1 Reaching Task
Uses the fixed_torso_environment with new XML file structure
"""
import numpy as np
import pickle
import argparse
import os
import sys
from collections import deque
import matplotlib.pyplot as plt

# Import the environment from fixed_torso_environment
from g1_rl_environment import G1ReachTouchEnv

# ============================================================================
# Helper Functions
# ============================================================================

def flatten_observation(obs):
    """Flatten observation to numpy array (handles dict or array observations)"""
    if isinstance(obs, dict):
        # Concatenate all values in the dict
        return np.concatenate([np.atleast_1d(v).flatten() for v in obs.values()])
    else:
        # Already an array
        return np.array(obs).flatten()


# ============================================================================
# Neural Network Policy Classes
# ============================================================================

class SimpleNeuralPolicy:
    """Simple feedforward neural network policy"""
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.params = None
    
    def init_params(self, seed=0):
        """Initialize network parameters with Xavier initialization"""
        np.random.seed(seed)
        
        self.params = {}
        layer_sizes = [self.obs_dim] + self.hidden_dims + [self.action_dim]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            self.params[f'w{i+1}'] = np.random.uniform(
                -limit, limit, (layer_sizes[i], layer_sizes[i+1])
            )
            self.params[f'b{i+1}'] = np.zeros(layer_sizes[i+1])
        
        return self.params
    
    def __call__(self, observation):
        """Forward pass through network"""
        if self.params is None:
            raise ValueError("Policy parameters not initialized!")
        
        x = observation
        num_layers = len([k for k in self.params.keys() if k.startswith('w')])
        
        for i in range(1, num_layers + 1):
            x = np.dot(x, self.params[f'w{i}']) + self.params[f'b{i}']
            if i < num_layers:  # ReLU for hidden layers
                x = np.maximum(0, x)
            else:  # Tanh for output layer
                x = np.tanh(x)
        
        return x
    
    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params


class ValueNetwork:
    """Value network for PPO (estimates state value)"""
    def __init__(self, obs_dim, hidden_dims=[256, 256]):
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        self.params = None
    
    def init_params(self, seed=0):
        """Initialize network parameters"""
        np.random.seed(seed)
        
        self.params = {}
        layer_sizes = [self.obs_dim] + self.hidden_dims + [1]
        
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            self.params[f'w{i+1}'] = np.random.uniform(
                -limit, limit, (layer_sizes[i], layer_sizes[i+1])
            )
            self.params[f'b{i+1}'] = np.zeros(layer_sizes[i+1])
        
        return self.params
    
    def __call__(self, observation):
        """Forward pass through network"""
        if self.params is None:
            raise ValueError("Value network parameters not initialized!")
        
        x = observation
        num_layers = len([k for k in self.params.keys() if k.startswith('w')])
        
        for i in range(1, num_layers + 1):
            x = np.dot(x, self.params[f'w{i}']) + self.params[f'b{i}']
            if i < num_layers:
                x = np.maximum(0, x)
        
        return x[0] if x.shape == (1,) else x.flatten()[0]
    
    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params


# ============================================================================
# PPO Helper Functions
# ============================================================================

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation (GAE)"""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return np.array(advantages)


def compute_policy_gradient_fast(policy, observations, actions, advantages):
    """
    Fast approximate policy gradient using finite differences
    Much faster than exact gradients for quick prototyping
    """
    epsilon = 1e-4
    params = policy.get_params()
    gradients = {}
    
    # Only compute gradients for a subset of parameters (sampling)
    param_keys = list(params.keys())
    
    for key in param_keys:
        param = params[key]
        grad = np.zeros_like(param)
        
        # Sample only a few indices for speed
        if param.size > 100:
            # For large matrices, sample randomly
            num_samples = min(50, param.size)
            flat_indices = np.random.choice(param.size, num_samples, replace=False)
        else:
            # For small matrices, compute all
            flat_indices = range(param.size)
        
        for idx in flat_indices:
            # Get multi-dimensional index
            multi_idx = np.unravel_index(idx, param.shape)
            
            # Finite difference
            original = param[multi_idx]
            
            param[multi_idx] = original + epsilon
            policy.set_params(params)
            loss_plus = 0
            for obs, act, adv in zip(observations, actions, advantages):
                pred_action = policy(obs)
                # Simple squared error weighted by advantage
                loss_plus += -adv * np.sum((pred_action - act) ** 2)
            
            param[multi_idx] = original - epsilon
            policy.set_params(params)
            loss_minus = 0
            for obs, act, adv in zip(observations, actions, advantages):
                pred_action = policy(obs)
                loss_minus += -adv * np.sum((pred_action - act) ** 2)
            
            # Restore and compute gradient
            param[multi_idx] = original
            grad[multi_idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        gradients[key] = grad
    
    policy.set_params(params)  # Restore original params
    return gradients


def apply_gradients(params, gradients, lr):
    """Apply gradients with learning rate"""
    new_params = {}
    for key in params.keys():
        new_params[key] = params[key] + lr * gradients[key]
    return new_params


# ============================================================================
# Main PPO Training Function
# ============================================================================

def train_ppo(scene_path="../unitree_g1/g1_table_box_scene.xml",
              num_epochs=30, 
              episodes_per_epoch=5, 
              max_steps=500,
              lr_policy=3e-4, 
              lr_value=1e-3, 
              gamma=0.99, 
              lam=0.95,
              clip_epsilon=0.2, 
              update_epochs=10, 
              minibatch_size=64,
              save_path=None):
    """
    Proximal Policy Optimization (PPO) for G1 Reaching Task
    
    Args:
        scene_path: Path to the MuJoCo XML scene file
        num_epochs: Number of training epochs
        episodes_per_epoch: Episodes to collect per epoch
        max_steps: Maximum steps per episode
        lr_policy: Learning rate for policy network
        lr_value: Learning rate for value network
        gamma: Discount factor
        lam: GAE lambda parameter
        clip_epsilon: PPO clipping parameter
        update_epochs: Number of update iterations per epoch
        minibatch_size: Size of minibatches for updates
        save_path: Path to save trained policy
    """
    
    print("\n" + "="*70)
    print("PPO TRAINING FOR G1 REACHING TASK")
    print("="*70)
    print(f"Scene Path: {scene_path}")
    print(f"Training epochs: {num_epochs}")
    print(f"Episodes per epoch: {episodes_per_epoch}")
    print(f"Max steps per episode: {max_steps}")
    print("="*70 + "\n")
    
    # Create environment
    env = G1ReachTouchEnv(scene_path=scene_path)
    
    # Get dimensions by doing a reset and checking the shapes
    obs = env.reset()
    obs = flatten_observation(obs)  # Flatten dict to array if needed
    obs_dim = len(obs)
    
    # Get action dimension from the environment's controllable actuators
    action_dim = len(env.controllable_actuators)
    
    print(f"Environment initialized:")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Controllable actuators: {action_dim}")
    print()
    
    # Initialize networks
    policy = SimpleNeuralPolicy(obs_dim, action_dim)
    policy.init_params(seed=42)
    
    value_net = ValueNetwork(obs_dim)
    value_net.init_params(seed=43)
    
    # Training tracking
    all_rewards = []
    all_distances = []
    all_success_rates = []
    best_reward = -float('inf')
    
    # Main training loop
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*70}")
        
        # Collect episodes
        epoch_observations = []
        epoch_actions = []
        epoch_rewards = []
        epoch_values = []
        epoch_dones = []
        epoch_next_values = []
        
        episode_rewards = []
        episode_distances = []
        episode_successes = []
        
        print(f"Collecting {episodes_per_epoch} episodes...")
        
        for ep in range(episodes_per_epoch):
            obs = env.reset()
            obs = flatten_observation(obs)  # Flatten dict to array
            
            episode_reward = 0
            episode_obs = []
            episode_acts = []
            episode_rews = []
            episode_vals = []
            episode_dones_list = []
            
            for step in range(max_steps):
                # Get action from policy
                action = policy(obs)
                value = value_net(obs)
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                next_obs = flatten_observation(next_obs)  # Flatten dict to array
                
                # Store transition
                episode_obs.append(obs)
                episode_acts.append(action)
                episode_rews.append(reward)
                episode_vals.append(value)
                episode_dones_list.append(1.0 if done else 0.0)
                
                episode_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            # Get final value for GAE computation
            final_value = value_net(obs) if not done else 0.0
            
            # Store episode data
            epoch_observations.extend(episode_obs)
            epoch_actions.extend(episode_acts)
            epoch_rewards.extend(episode_rews)
            epoch_values.extend(episode_vals)
            epoch_dones.extend(episode_dones_list)
            epoch_next_values.extend(episode_vals[1:] + [final_value])
            
            episode_rewards.append(episode_reward)
            episode_distances.append(info.get('distance', 0))
            episode_successes.append(1 if info.get('success', False) else 0)
            
            print(f"  Episode {ep+1}/{episodes_per_epoch}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Steps={len(episode_obs)}, "
                  f"Distance={info.get('distance', 0):.4f}m")
        
        # Compute advantages
        print(f"\nCollected {len(epoch_observations)} timesteps. Computing advantages...")
        advantages = compute_gae(
            epoch_rewards, epoch_values, epoch_next_values, 
            epoch_dones, gamma, lam
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to arrays
        observations = np.array(epoch_observations)
        actions = np.array(epoch_actions)
        returns = advantages + np.array(epoch_values)
        
        # PPO update
        print("Starting PPO updates...")
        dataset_size = len(observations)
        
        for update_epoch in range(update_epochs):
            # Shuffle data
            indices = np.random.permutation(dataset_size)
            
            num_batches = (dataset_size + minibatch_size - 1) // minibatch_size
            print(f"  Update epoch {update_epoch+1}/{update_epochs}: "
                  f"Processing {num_batches} minibatches...", end='', flush=True)
            
            # Process in minibatches
            for start in range(0, dataset_size, minibatch_size):
                end = min(start + minibatch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_acts = actions[batch_indices]
                batch_advs = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Update policy (fast approximate gradients)
                policy_grads = compute_policy_gradient_fast(
                    policy, batch_obs, batch_acts, batch_advs
                )
                policy_params = policy.get_params()
                policy_params = apply_gradients(policy_params, policy_grads, lr_policy)
                policy.set_params(policy_params)
                
                # Update value network (simple MSE gradient)
                value_grads = {}
                value_params = value_net.get_params()
                
                for key in value_params.keys():
                    grad = np.zeros_like(value_params[key])
                    
                    # Compute gradient for value loss
                    for obs, ret in zip(batch_obs, batch_returns):
                        value_pred = value_net(obs)
                        error = value_pred - ret
                        
                        # Simple gradient approximation
                        eps = 1e-4
                        original = value_params[key].copy()
                        
                        # Sample a few parameters for speed
                        if value_params[key].size > 50:
                            sample_size = 20
                        else:
                            sample_size = value_params[key].size
                        
                        flat_indices = np.random.choice(
                            value_params[key].size, 
                            min(sample_size, value_params[key].size), 
                            replace=False
                        )
                        
                        for idx in flat_indices:
                            multi_idx = np.unravel_index(idx, value_params[key].shape)
                            value_params[key][multi_idx] += eps
                            value_net.set_params(value_params)
                            loss_plus = (value_net(obs) - ret) ** 2
                            
                            value_params[key][multi_idx] = original[multi_idx] - eps
                            value_net.set_params(value_params)
                            loss_minus = (value_net(obs) - ret) ** 2
                            
                            grad[multi_idx] += (loss_plus - loss_minus) / (2 * eps)
                            value_params[key][multi_idx] = original[multi_idx]
                    
                    value_grads[key] = grad / len(batch_obs)
                
                value_params = apply_gradients(value_params, value_grads, -lr_value)
                value_net.set_params(value_params)
            
            print(" Done!")
        
        # Log epoch results
        avg_reward = np.mean(episode_rewards)
        avg_distance = np.mean(episode_distances)
        success_rate = np.mean(episode_successes) * 100
        
        all_rewards.append(avg_reward)
        all_distances.append(avg_distance)
        all_success_rates.append(success_rate)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Distance: {avg_distance:.4f}m")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            print(f"  🎉 NEW BEST REWARD: {best_reward:.2f}")
    
    # Save results
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    if save_path:
        results = {
            'policy_params': policy.get_params(),
            'value_params': value_net.get_params(),
            'rewards': all_rewards,
            'distances': all_distances,
            'success_rates': all_success_rates,
            'config': {
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'num_epochs': num_epochs,
                'episodes_per_epoch': episodes_per_epoch,
                'max_steps': max_steps
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✅ Results saved to: {save_path}")
    
    # Plot training progress
    plot_training_progress(all_rewards, all_distances, all_success_rates)
    
    return policy, value_net, all_rewards


def plot_training_progress(rewards, distances, success_rates):
    """Plot training progress"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rewards
    axes[0].plot(rewards, 'b-', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Training Rewards')
    axes[0].grid(True, alpha=0.3)
    
    # Distances
    axes[1].plot(distances, 'r-', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Distance (m)')
    axes[1].set_title('Distance to Target')
    axes[1].grid(True, alpha=0.3)
    
    # Success rates
    axes[2].plot(success_rates, 'g-', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_title('Success Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppo_training_progress.png', dpi=150)
    print(f"\n📊 Training plot saved to: ppo_training_progress.png")
    plt.close()


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Training for G1 Reaching')
    parser.add_argument('--scene', type=str, 
                       default="../unitree_g1/g1_table_box_scene.xml",
                       help='Path to scene XML file')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--episodes_per_epoch', type=int, default=5,
                       help='Episodes per epoch')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Max steps per episode')
    parser.add_argument('--save', type=str, default='ppo_g1_policy.pkl',
                       help='Path to save trained policy')
    parser.add_argument('--lr_policy', type=float, default=3e-4,
                       help='Policy learning rate')
    parser.add_argument('--lr_value', type=float, default=1e-3,
                       help='Value network learning rate')
    
    args = parser.parse_args()
    
    # Verify scene file exists
    if not os.path.exists(args.scene):
        print(f"❌ Error: Scene file not found at {args.scene}")
        print("\nPlease check the path. Expected structure:")
        print("  fixed_torso_environment/")
        print("    ├── ppo_training_script.py  (this file)")
        print("    ├── g1_rl_environment.py")
        print("    └── ../unitree_g1/g1_table_box_scene.xml")
        sys.exit(1)
    
    # Run training
    policy, value_net, rewards = train_ppo(
        scene_path=args.scene,
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        max_steps=args.max_steps,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        save_path=args.save
    )
    
    print("\n✨ Training completed successfully!")