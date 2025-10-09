"""
Training script for G1 Reaching Environment
Includes Random, Evolutionary Strategy, and PPO algorithms
"""
import numpy as np
import pickle
import argparse
from collections import deque
from g1_reaching_env import G1ReachingEnv

# ============================================================================
# Policy Classes
# ============================================================================

class RandomPolicy:
    """Random policy for baseline testing"""
    def __init__(self, action_dim):
        self.action_dim = action_dim
    
    def __call__(self, observation):
        return np.random.uniform(-1.0, 1.0, size=self.action_dim)
    
    def get_params(self):
        return None


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
            raise ValueError("Policy parameters not initialized! Call init_params() first.")
        
        x = observation
        num_layers = len([k for k in self.params.keys() if k.startswith('w')])
        
        for i in range(1, num_layers):
            x = np.dot(x, self.params[f'w{i}']) + self.params[f'b{i}']
            x = np.tanh(x)
        
        # Final layer without activation, then tanh to bound actions
        x = np.dot(x, self.params[f'w{num_layers}']) + self.params[f'b{num_layers}']
        x = np.tanh(x)
        
        return x
    
    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params


class ValueNetwork:
    """Value function approximator for PPO"""
    def __init__(self, obs_dim, hidden_dims=[256, 256]):
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        self.params = None
    
    def init_params(self, seed=0):
        """Initialize network parameters"""
        np.random.seed(seed + 1000)
        
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
        """Forward pass to get value estimate"""
        if self.params is None:
            raise ValueError("Value network not initialized!")
        
        x = observation
        num_layers = len([k for k in self.params.keys() if k.startswith('w')])
        
        for i in range(1, num_layers):
            x = np.dot(x, self.params[f'w{i}']) + self.params[f'b{i}']
            x = np.tanh(x)
        
        # Final layer - single value output
        x = np.dot(x, self.params[f'w{num_layers}']) + self.params[f'b{num_layers}']
        
        return x[0] if x.shape == (1,) else x
    
    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params


# ============================================================================
# Training Functions
# ============================================================================

def run_episode(env, policy, max_steps=1000, seed=None):
    """Run a single episode and collect data"""
    obs = env.reset(seed=seed)
    
    observations = []
    actions = []
    rewards = []
    infos = []
    
    for step in range(max_steps):
        # Get action from policy
        action = policy(obs)
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        # Store data
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        
        obs = next_obs
        
        if done:
            break
    
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'infos': infos,
        'total_reward': np.sum(rewards),
        'episode_length': len(rewards)
    }


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0
    
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * last_advantage
    
    return advantages


def train_random_policy(num_episodes=10, max_steps=1000):
    """Train/test with random policy (baseline)"""
    print("Testing random policy...")
    print("="*60)
    
    env = G1ReachingEnv(max_episode_steps=max_steps)
    policy = RandomPolicy(env.action_space_dim)
    
    results = []
    
    for episode in range(num_episodes):
        trajectory = run_episode(env, policy, max_steps=max_steps, seed=episode)
        
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Total Reward: {trajectory['total_reward']:.2f}")
        print(f"  Episode Length: {trajectory['episode_length']}")
        print(f"  Final Distance: {trajectory['infos'][-1]['distance']:.4f}m")
        
        results.append(trajectory)
    
    avg_reward = np.mean([r['total_reward'] for r in results])
    print("="*60)
    print(f"Average Reward: {avg_reward:.2f}")
    
    return results, policy


def train_evolutionary_strategy(num_episodes=100, max_steps=1000, 
                                learning_rate=0.1, population_size=10):
    """
    Improved evolutionary strategy with better exploration
    """
    print("Training with Evolutionary Strategy (Improved)...")
    print("="*60)
    
    env = G1ReachingEnv(max_episode_steps=max_steps)
    
    # Initialize policy with larger network
    policy = SimpleNeuralPolicy(
        obs_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
        hidden_dims=[256, 256]
    )
    policy.init_params(seed=42)
    
    best_params = policy.get_params()
    best_reward = -float('inf')
    recent_rewards = deque(maxlen=10)
    
    # Adaptive learning rate
    current_lr = learning_rate
    
    for episode in range(num_episodes):
        # Generate population of policies
        candidates = []
        for i in range(population_size):
            # Add Gaussian noise to best parameters
            noisy_params = {}
            for key in best_params:
                noise = np.random.randn(*best_params[key].shape) * current_lr
                noisy_params[key] = best_params[key] + noise
            
            # Evaluate candidate
            policy.set_params(noisy_params)
            trajectory = run_episode(env, policy, max_steps=max_steps, seed=episode*100+i)
            
            candidates.append({
                'params': noisy_params,
                'reward': trajectory['total_reward']
            })
        
        # Select top performers
        candidates.sort(key=lambda x: x['reward'], reverse=True)
        top_k = max(1, population_size // 3)
        
        # Average top performers (simple crossover)
        avg_params = {}
        for key in best_params:
            avg_params[key] = np.mean([c['params'][key] for c in candidates[:top_k]], axis=0)
        
        # Update if better
        best_candidate = candidates[0]
        if best_candidate['reward'] > best_reward:
            best_reward = best_candidate['reward']
            best_params = avg_params  # Use average of top performers
            print(f"Episode {episode + 1}: NEW BEST Reward = {best_reward:.2f}")
        else:
            print(f"Episode {episode + 1}: Best = {best_reward:.2f}, Current = {best_candidate['reward']:.2f}")
        
        recent_rewards.append(best_candidate['reward'])
        
        # Adaptive learning rate
        if len(recent_rewards) >= 10:
            if np.std(recent_rewards) < 5.0:  # Converged, explore more
                current_lr = min(current_lr * 1.1, learning_rate * 2)
            else:
                current_lr = max(current_lr * 0.95, learning_rate * 0.1)
    
    policy.set_params(best_params)
    return policy


def train_ppo(num_epochs=100, episodes_per_epoch=5, max_steps=500,
              lr_policy=3e-4, lr_value=1e-3, gamma=0.99, lam=0.95,
              clip_epsilon=0.2, update_epochs=10, minibatch_size=64):
    """
    Proximal Policy Optimization (PPO) - memory efficient version
    Designed to work on limited hardware
    """
    print("Training with PPO...")
    print("="*60)
    print(f"Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Episodes per epoch: {episodes_per_epoch}")
    print(f"  Max steps: {max_steps}")
    print(f"  Policy LR: {lr_policy}")
    print(f"  Value LR: {lr_value}")
    print(f"  Clip ε: {clip_epsilon}")
    print("="*60)
    
    env = G1ReachingEnv(max_episode_steps=max_steps)
    
    # Initialize policy and value networks
    policy = SimpleNeuralPolicy(
        obs_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
        hidden_dims=[256, 256]
    )
    policy.init_params(seed=42)
    
    value_net = ValueNetwork(
        obs_dim=env.observation_space_dim,
        hidden_dims=[256, 256]
    )
    value_net.init_params(seed=43)
    
    best_reward = -float('inf')
    best_policy_params = policy.get_params()
    
    for epoch in range(num_epochs):
        # Collect trajectories
        all_observations = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_old_log_probs = []
        
        epoch_rewards = []
        
        for ep in range(episodes_per_epoch):
            trajectory = run_episode(env, policy, max_steps=max_steps, seed=epoch*100+ep)
            epoch_rewards.append(trajectory['total_reward'])
            
            observations = trajectory['observations']
            actions = trajectory['actions']
            rewards = trajectory['rewards']
            
            # Compute values
            values = np.array([value_net(obs) for obs in observations])
            
            # Compute advantages using GAE
            advantages = compute_gae(rewards, values, gamma, lam)
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute old log probabilities (Gaussian policy)
            old_log_probs = -0.5 * np.sum(actions**2, axis=-1)
            
            all_observations.extend(observations)
            all_actions.extend(actions)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            all_old_log_probs.extend(old_log_probs)
        
        # Convert to arrays
        all_observations = np.array(all_observations)
        all_actions = np.array(all_actions)
        all_advantages = np.array(all_advantages)
        all_returns = np.array(all_returns)
        all_old_log_probs = np.array(all_old_log_probs)
        
        # PPO update
        dataset_size = len(all_observations)
        
        for update_epoch in range(update_epochs):
            # Shuffle data
            indices = np.random.permutation(dataset_size)
            
            # Process in minibatches
            for start in range(0, dataset_size, minibatch_size):
                end = min(start + minibatch_size, dataset_size)
                batch_indices = indices[start:end]
                
                obs_batch = all_observations[batch_indices]
                actions_batch = all_actions[batch_indices]
                advantages_batch = all_advantages[batch_indices]
                returns_batch = all_returns[batch_indices]
                old_log_probs_batch = all_old_log_probs[batch_indices]
                
                # Update policy
                policy_params = policy.get_params()
                policy_grad = compute_policy_gradient(
                    policy, obs_batch, actions_batch, advantages_batch,
                    old_log_probs_batch, clip_epsilon
                )
                
                # Apply gradient descent
                for key in policy_params:
                    policy_params[key] -= lr_policy * policy_grad[key]
                
                policy.set_params(policy_params)
                
                # Update value network
                value_params = value_net.get_params()
                value_grad = compute_value_gradient(
                    value_net, obs_batch, returns_batch
                )
                
                # Apply gradient descent
                for key in value_params:
                    value_params[key] -= lr_value * value_grad[key]
                
                value_net.set_params(value_params)
        
        # Log progress
        avg_reward = np.mean(epoch_rewards)
        max_reward = np.max(epoch_rewards)
        
        if max_reward > best_reward:
            best_reward = max_reward
            best_policy_params = policy.get_params()
            print(f"Epoch {epoch + 1}/{num_epochs}: NEW BEST! Avg={avg_reward:.2f}, Max={max_reward:.2f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}: Avg={avg_reward:.2f}, Max={max_reward:.2f}, Best={best_reward:.2f}")
    
    # Restore best policy
    policy.set_params(best_policy_params)
    return policy


def compute_policy_gradient(policy, observations, actions, advantages, 
                            old_log_probs, clip_epsilon):
    """Compute PPO policy gradient (finite differences approximation)"""
    params = policy.get_params()
    gradients = {key: np.zeros_like(params[key]) for key in params}
    
    epsilon = 1e-5
    
    for key in params:
        # Flatten parameter
        param_flat = params[key].flatten()
        grad_flat = np.zeros_like(param_flat)
        
        # Compute gradient using finite differences (only sample a subset for efficiency)
        sample_size = min(len(param_flat), 100)
        sample_indices = np.random.choice(len(param_flat), sample_size, replace=False)
        
        for idx in sample_indices:
            # Perturb parameter
            param_flat[idx] += epsilon
            params[key] = param_flat.reshape(params[key].shape)
            policy.set_params(params)
            
            # Compute loss with perturbation
            loss_plus = compute_ppo_loss(policy, observations, actions, advantages, 
                                        old_log_probs, clip_epsilon)
            
            # Restore and perturb in opposite direction
            param_flat[idx] -= 2 * epsilon
            params[key] = param_flat.reshape(params[key].shape)
            policy.set_params(params)
            
            loss_minus = compute_ppo_loss(policy, observations, actions, advantages,
                                         old_log_probs, clip_epsilon)
            
            # Compute gradient
            grad_flat[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore parameter
            param_flat[idx] += epsilon
        
        gradients[key] = grad_flat.reshape(params[key].shape)
    
    # Restore original parameters
    policy.set_params(params)
    
    return gradients


def compute_value_gradient(value_net, observations, returns):
    """Compute value network gradient (finite differences)"""
    params = value_net.get_params()
    gradients = {key: np.zeros_like(params[key]) for key in params}
    
    epsilon = 1e-5
    
    for key in params:
        param_flat = params[key].flatten()
        grad_flat = np.zeros_like(param_flat)
        
        # Sample subset for efficiency
        sample_size = min(len(param_flat), 100)
        sample_indices = np.random.choice(len(param_flat), sample_size, replace=False)
        
        for idx in sample_indices:
            # Perturb parameter
            param_flat[idx] += epsilon
            params[key] = param_flat.reshape(params[key].shape)
            value_net.set_params(params)
            
            # Compute loss
            loss_plus = compute_value_loss(value_net, observations, returns)
            
            param_flat[idx] -= 2 * epsilon
            params[key] = param_flat.reshape(params[key].shape)
            value_net.set_params(params)
            
            loss_minus = compute_value_loss(value_net, observations, returns)
            
            # Gradient
            grad_flat[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore
            param_flat[idx] += epsilon
        
        gradients[key] = grad_flat.reshape(params[key].shape)
    
    value_net.set_params(params)
    return gradients


def compute_ppo_loss(policy, observations, actions, advantages, old_log_probs, clip_epsilon):
    """Compute PPO clipped objective"""
    # Compute new log probs (simplified Gaussian policy)
    new_log_probs = []
    for obs, action in zip(observations, actions):
        pred_action = policy(obs)
        log_prob = -0.5 * np.sum((action - pred_action)**2)
        new_log_probs.append(log_prob)
    
    new_log_probs = np.array(new_log_probs)
    
    # Compute ratio
    ratio = np.exp(new_log_probs - old_log_probs)
    
    # Clipped objective
    clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))
    
    return loss


def compute_value_loss(value_net, observations, returns):
    """Compute value function MSE loss"""
    predictions = np.array([value_net(obs) for obs in observations])
    loss = np.mean((predictions - returns)**2)
    return loss


def save_policy(policy, filename='trained_policy.pkl'):
    """Save trained policy to disk"""
    data = {
        'policy_type': type(policy).__name__,
        'params': policy.get_params(),
        'obs_dim': policy.obs_dim if hasattr(policy, 'obs_dim') else None,
        'action_dim': policy.action_dim if hasattr(policy, 'action_dim') else None,
        'hidden_dims': policy.hidden_dims if hasattr(policy, 'hidden_dims') else None,
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Policy saved to {filename}")


def load_policy(filename='trained_policy.pkl'):
    """Load trained policy from disk"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    if data['policy_type'] == 'SimpleNeuralPolicy':
        policy = SimpleNeuralPolicy(
            obs_dim=data['obs_dim'],
            action_dim=data['action_dim'],
            hidden_dims=data['hidden_dims']
        )
        policy.set_params(data['params'])
    else:
        raise ValueError(f"Unknown policy type: {data['policy_type']}")
    
    print(f"Policy loaded from {filename}")
    return policy



# Main code for training
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train G1 reaching policy')
    parser.add_argument('--mode', type=str, default='ppo',
                        choices=['random', 'es', 'ppo'],
                        help='Training algorithm: random, es (evolutionary strategy), or ppo')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes (for random/es) or epochs (for ppo)')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--save', type=str, default=None,
                        help='Filename to save trained policy')
    parser.add_argument('--load', type=str, default=None,
                        help='Filename to load pretrained policy')
    
    # ES params
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for evolutionary strategy')
    parser.add_argument('--population_size', type=int, default=10,
                        help='Population size for evolutionary strategy')
    
    # PPO params
    parser.add_argument('--episodes_per_epoch', type=int, default=5,
                        help='Episodes per epoch for PPO')
    parser.add_argument('--lr_policy', type=float, default=3e-4,
                        help='Policy learning rate for PPO')
    parser.add_argument('--lr_value', type=float, default=1e-3,
                        help='Value learning rate for PPO')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO clipping parameter')
    
    args = parser.parse_args()
    
    if args.load:
        # Load and test a pretrained policy
        policy = load_policy(args.load)
        env = G1ReachingEnv(max_episode_steps=args.max_steps)
        
        print("\nTesting loaded policy...")
        trajectory = run_episode(env, policy, max_steps=args.max_steps)
        print(f"Total Reward: {trajectory['total_reward']:.2f}")
        print(f"Episode Length: {trajectory['episode_length']}")
        print(f"Final Distance: {trajectory['infos'][-1]['distance']:.4f}m")
    
    elif args.mode == 'random':
        results, policy = train_random_policy(
            num_episodes=args.episodes,
            max_steps=args.max_steps
        )
    
    elif args.mode == 'es':
        policy = train_evolutionary_strategy(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            population_size=args.population_size
        )
        
        if args.save:
            save_policy(policy, args.save)
        
        print("\nTraining complete!")
    
    elif args.mode == 'ppo':
        policy = train_ppo(
            num_epochs=args.episodes,  # Reuse episodes arg as epochs
            episodes_per_epoch=args.episodes_per_epoch,
            max_steps=args.max_steps,
            lr_policy=args.lr_policy,
            lr_value=args.lr_value,
            clip_epsilon=args.clip_epsilon
        )
        
        if args.save:
            save_policy(policy, args.save)
        
        print("\nPPO Training complete!")
        print("Run visualization with:")
        print(f"  python g1_reaching_visualize.py --mode trained --policy {args.save or 'trained_policy.pkl'}")