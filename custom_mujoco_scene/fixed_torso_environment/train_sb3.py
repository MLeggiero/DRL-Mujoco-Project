#!/usr/bin/env python3
"""
Unitree G1 Reaching Task Training with Stable-Baselines3 PPO.

Production-ready reinforcement learning training script with configurable
hyperparameters and parallel environment support.
"""
import os
import argparse
import numpy as np
from datetime import datetime
import torch
import matplotlib
# Detect display availability (for WSL/headless environments)
if os.environ.get('DISPLAY') is None:
    print("No DISPLAY detected - using Agg backend (will save plots to file)")
    matplotlib.use('Agg')
    DISPLAY_AVAILABLE = False
else:
    try:
        matplotlib.use('TkAgg')  # Try TkAgg for interactive plotting
        DISPLAY_AVAILABLE = True
    except:
        print("TkAgg backend failed - using Agg backend (will save plots to file)")
        matplotlib.use('Agg')
        DISPLAY_AVAILABLE = False
import matplotlib.pyplot as plt
from collections import deque

# Import Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# Import our custom environment
from g1_gym_wrapper import G1ReachingGymEnv


class TrainingProgressCallback:
    """Custom callback for monitoring training progress.

    Tracks episode statistics and prints periodic progress updates.
    """

    def __init__(self, check_freq=1000):
        self.check_freq = check_freq
        self.n_calls = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []

    def __call__(self, locals_dict, globals_dict):
        self.n_calls += 1

        # Get info from the environment
        infos = locals_dict.get('infos', [])

        for info in infos:
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

            if 'distance_to_target' in info:
                self.episode_distances.append(info['distance_to_target'])

        # Print progress every N steps
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_distance = np.mean(self.episode_distances[-100:]) if self.episode_distances else 0

                print(f"Steps: {self.n_calls:>7,} | "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"Avg Length: {avg_length:>5.0f} | "
                      f"Avg Distance: {avg_distance:.3f}m")

        return True


class DynamicPlottingCallback(BaseCallback):
    """
    Callback for creating dynamic real-time plots of training progress.

    Displays episode rewards vs environment steps with a moving average.
    If no display is available, saves plots to files instead.
    """

    def __init__(self, plot_freq=100, window_size=100, save_dir="./plots", verbose=0):
        """
        Args:
            plot_freq: Update plot every N steps
            window_size: Size of moving average window
            save_dir: Directory to save plots (when display unavailable)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.window_size = window_size
        self.save_dir = save_dir
        self.display_available = DISPLAY_AVAILABLE

        # Data storage
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_distances = []

        # For moving average
        self.reward_buffer = deque(maxlen=window_size)
        self.distance_buffer = deque(maxlen=window_size)

        # Plotting setup
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line_rewards = None
        self.line_avg_rewards = None
        self.line_distances = None
        self.line_avg_distances = None

        # Create save directory if needed
        if not self.display_available:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Plot files will be saved to: {save_dir}")
        else:
            # Interactive mode only if display available
            plt.ion()

    def _on_training_start(self):
        """Initialize the plot when training starts."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Training Progress', fontsize=14, fontweight='bold')

        # Rewards plot
        self.ax1.set_xlabel('Environment Steps')
        self.ax1.set_ylabel('Episode Reward')
        self.ax1.set_title('Episode Rewards')
        self.ax1.grid(True, alpha=0.3)
        self.line_rewards, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw Rewards')
        self.line_avg_rewards, = self.ax1.plot([], [], 'r-', linewidth=2, label=f'Moving Avg ({self.window_size})')
        self.ax1.legend()

        # Distance plot
        self.ax2.set_xlabel('Environment Steps')
        self.ax2.set_ylabel('Distance to Target (m)')
        self.ax2.set_title('Final Episode Distance')
        self.ax2.grid(True, alpha=0.3)
        self.line_distances, = self.ax2.plot([], [], 'g-', alpha=0.3, label='Raw Distance')
        self.line_avg_distances, = self.ax2.plot([], [], 'orange', linewidth=2, label=f'Moving Avg ({self.window_size})')
        self.ax2.axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='Success Threshold (5cm)')
        self.ax2.legend()

        plt.tight_layout()

        if self.display_available:
            plt.show(block=False)
            plt.pause(0.001)

    def _on_step(self):
        """Called at every step."""
        # Check if episode finished
        for idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                # Get episode info from the Monitor wrapper
                infos = self.locals.get('infos', [])
                if idx < len(infos) and 'episode' in infos[idx]:
                    episode_reward = infos[idx]['episode']['r']
                    episode_length = infos[idx]['episode']['l']
                    current_step = self.num_timesteps

                    # Store episode data
                    self.episode_rewards.append(episode_reward)
                    self.episode_steps.append(current_step)
                    self.reward_buffer.append(episode_reward)

                    # Get distance if available
                    if 'distance_to_target' in infos[idx]:
                        distance = infos[idx]['distance_to_target']
                        self.episode_distances.append(distance)
                        self.distance_buffer.append(distance)

        # Update plot periodically
        if self.num_timesteps % self.plot_freq == 0 and len(self.episode_rewards) > 0:
            self._update_plot()

        return True

    def _update_plot(self):
        """Update the dynamic plot with latest data."""
        if self.fig is None:
            return

        try:
            # Update rewards plot
            self.line_rewards.set_data(self.episode_steps, self.episode_rewards)

            # Calculate and plot moving average
            if len(self.episode_rewards) > 0:
                avg_rewards = []
                avg_steps = []
                for i in range(len(self.episode_rewards)):
                    start_idx = max(0, i - self.window_size + 1)
                    avg_rewards.append(np.mean(self.episode_rewards[start_idx:i+1]))
                    avg_steps.append(self.episode_steps[i])
                self.line_avg_rewards.set_data(avg_steps, avg_rewards)

            # Update distances plot
            if len(self.episode_distances) > 0:
                distance_steps = self.episode_steps[-len(self.episode_distances):]
                self.line_distances.set_data(distance_steps, self.episode_distances)

                # Calculate and plot moving average for distances
                avg_distances = []
                avg_dist_steps = []
                for i in range(len(self.episode_distances)):
                    start_idx = max(0, i - self.window_size + 1)
                    avg_distances.append(np.mean(self.episode_distances[start_idx:i+1]))
                    avg_dist_steps.append(distance_steps[i])
                self.line_avg_distances.set_data(avg_dist_steps, avg_distances)

            # Rescale axes
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()

            if self.display_available:
                # Redraw for interactive display
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.001)
            else:
                # Save to file instead
                save_path = os.path.join(self.save_dir, f"training_progress_step_{self.num_timesteps}.png")
                self.fig.savefig(save_path, dpi=100, bbox_inches='tight')
                if self.verbose > 0:
                    print(f"Plot saved to: {save_path}")

        except Exception as e:
            if self.verbose > 0:
                print(f"Plot update error: {e}")

    def _on_training_end(self):
        """Save final plot and optionally keep it open."""
        if self.fig is not None:
            # Always save final plot
            final_path = os.path.join(self.save_dir, "training_progress_final.png")
            os.makedirs(self.save_dir, exist_ok=True)
            self.fig.savefig(final_path, dpi=150, bbox_inches='tight')
            print(f"\nFinal training plot saved to: {final_path}")

            if self.display_available:
                plt.ioff()
                print("Close the plot window to exit.")
                # Keep plot open
                plt.show()
            else:
                plt.close(self.fig)


def make_env(scene_path, rank=0, seed=0, action_smoothing=0.3, smoothness_weight=1.0,
             action_scale=0.25, sim_substeps=10):
    """Create and wrap the environment with smoothness parameters.

    Args:
        scene_path: Path to MuJoCo scene XML file
        rank: Environment rank for parallel training
        seed: Random seed for reproducibility
        action_smoothing: EMA coefficient for action filtering
        smoothness_weight: Weight for smoothness penalty in reward
        action_scale: Scaling factor for actuator commands
        sim_substeps: Physics substeps per RL step

    Returns:
        Callable that initializes the environment
    """
    def _init():
        env = G1ReachingGymEnv(
            scene_path=scene_path,
            action_smoothing=action_smoothing,
            smoothness_weight=smoothness_weight,
            action_scale=action_scale,
            sim_substeps=sim_substeps
        )
        env.reset(seed=seed + rank)
        # Wrap with Monitor to track episode statistics
        env = Monitor(env)
        return env
    return _init


def train_g1_reaching(
    scene_path="../unitree_g1/g1_table_box_scene.xml",
    total_timesteps=100_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    save_freq=10_000,
    eval_freq=5_000,
    log_dir="./logs",
    model_dir="./models",
    seed=0,
    device='cuda',
    action_smoothing=0.3,
    smoothness_weight=1.0,
    action_scale=0.25,
    sim_substeps=10,
    n_envs=4,
    enable_plot=True,
):
    """Train G1 reaching policy using Stable-Baselines3 PPO.

    Args:
        scene_path: Path to MuJoCo scene XML
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps to collect per update
        batch_size: Minibatch size for PPO updates
        n_epochs: Number of epochs for PPO updates
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        save_freq: Save model every N steps
        eval_freq: Evaluate model every N steps
        log_dir: Directory for logs
        model_dir: Directory for saved models
        seed: Random seed
        device: Device to use (cuda/cpu/auto)
        action_smoothing: EMA coefficient for action filtering
        smoothness_weight: Weight for action smoothness penalty
        action_scale: Scaling factor for actions
        sim_substeps: Number of simulation steps per RL step
        n_envs: Number of parallel environments
        enable_plot: Enable real-time plotting of training progress

    Returns:
        Tuple of (trained model, vectorized environment)
    """

    # Check CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    elif device == 'cuda':
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    print("=" * 70)
    print("Stable-Baselines3 PPO Training - G1 Reaching Task")
    print("=" * 70)
    print(f"Scene: {scene_path}")
    print(f"Device: {device.upper()}")
    print(f"Parallel Environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Rollout steps: {n_steps}")
    print(f"Epochs: {n_epochs}")
    print(f"Gamma: {gamma}")
    print("=" * 70)
    print()

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"g1_ppo_{timestamp}"

    # Create the training environment with smoothness parameters
    print(f"Creating {n_envs} parallel training environments...")

    # Use SubprocVecEnv for parallel environments (much faster!)
    # SubprocVecEnv runs each environment in a separate process
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(
                scene_path, rank=i, seed=seed + i,
                action_smoothing=action_smoothing,
                smoothness_weight=smoothness_weight,
                action_scale=action_scale,
                sim_substeps=sim_substeps
            ) for i in range(n_envs)
        ], start_method='fork')  # 'fork' is faster on Linux
    else:
        # Single environment - use DummyVecEnv
        env = DummyVecEnv([make_env(
            scene_path, rank=0, seed=seed,
            action_smoothing=action_smoothing,
            smoothness_weight=smoothness_weight,
            action_scale=action_scale,
            sim_substeps=sim_substeps
        )])

    print(f"[OK] Training environments created (using {'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'})")

    # Normalize observations and rewards for better training stability
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=500.0,  # Increased to preserve MASSIVE success bonus (up to +2000)
        gamma=gamma
    )

    # Create evaluation environment (single env for consistent evaluation)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(
        scene_path, rank=n_envs, seed=seed + 10000,
        action_smoothing=action_smoothing,
        smoothness_weight=smoothness_weight,
        action_scale=action_scale,
        sim_substeps=sim_substeps
    )])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during evaluation
        clip_obs=10.0,
        training=False
    )

    # Setup callbacks
    # Note: save_freq and eval_freq are per environment, so divide by n_envs
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=f"{model_dir}/{run_name}",
        name_prefix="g1_ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/{run_name}/best_model",
        log_path=f"{log_dir}/{run_name}",
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Build callback list
    callbacks = [checkpoint_callback, eval_callback]

    # Add dynamic plotting callback if enabled
    if enable_plot:
        plot_save_dir = os.path.join(log_dir, run_name, "plots")
        plot_callback = DynamicPlottingCallback(
            plot_freq=500,  # Update plot every 500 steps
            window_size=100,  # Moving average window of 100 episodes
            save_dir=plot_save_dir,
            verbose=0
        )
        callbacks.append(plot_callback)
        if DISPLAY_AVAILABLE:
            print("Dynamic plotting enabled - interactive graphs will display during training")
        else:
            print(f"Dynamic plotting enabled - plots will be saved to {plot_save_dir}")

    callback = CallbackList(callbacks)

    # Create PPO model with improved hyperparameters for smoother control
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0005,  # REDUCED from 0.01 to reduce exploration noise and shakiness
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=0.02,  # CRITICAL: Stop updates early if policy changes too much (prevents instability)
        tensorboard_log=f"{log_dir}/{run_name}",
        policy_kwargs=dict(
            # INCREASED network size from 256-256 to 512-512 for smoother, more capable policies
            net_arch=[dict(pi=[512, 512], vf=[512, 512])],
            # Use tanh activation for smoother outputs
            activation_fn=torch.nn.Tanh
        ),
        verbose=1,
        seed=seed,
        device=device
    )

    print("\nModel architecture:")
    print(model.policy)
    print()

    # Train the model
    print("="*70)
    print("Starting training...")
    print("="*70)
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True
        )

        print()
        print("="*70)
        print("[SUCCESS] Training complete!")
        print("="*70)

        # Save final model
        final_model_path = f"{model_dir}/{run_name}/final_model"
        model.save(final_model_path)
        env.save(f"{final_model_path}_vecnormalize.pkl")

        print(f"\nFinal model saved to: {final_model_path}")
        print(f"Best model saved to: {model_dir}/{run_name}/best_model")
        print(f"\nTo visualize training progress, run:")
        print(f"  tensorboard --logdir {log_dir}/{run_name}")

        return model, env

    except KeyboardInterrupt:
        print()
        print("="*70)
        print("Training interrupted by user")
        print("="*70)

        # Save interrupted model
        interrupt_path = f"{model_dir}/{run_name}/interrupted_model"
        model.save(interrupt_path)
        env.save(f"{interrupt_path}_vecnormalize.pkl")
        print(f"\nInterrupted model saved to: {interrupt_path}")

        return model, env


def test_trained_model(model_path, scene_path, n_episodes=5):
    """Test a trained model"""
    print("="*70)
    print("TESTING TRAINED MODEL")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("="*70)
    print()

    # Load the model
    model = PPO.load(model_path)

    # Load the normalization stats
    vecnormalize_path = f"{model_path}_vecnormalize.pkl"
    if os.path.exists(vecnormalize_path):
        print("Loading normalization statistics...")

    # Create test environment
    env = DummyVecEnv([make_env(scene_path, rank=0, seed=42)])

    if os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False

    # Test the model
    episode_rewards = []
    episode_distances = []
    episode_successes = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        final_distance = 0

        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1

            if len(info) > 0 and 'distance_to_target' in info[0]:
                final_distance = info[0]['distance_to_target']

        success = final_distance < 0.05
        episode_rewards.append(episode_reward)
        episode_distances.append(final_distance)
        episode_successes.append(success)

        print(f"Episode {episode+1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Distance={final_distance:.3f}m, "
              f"Success={success}")

    print()
    print("="*70)
    print("Test Results:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Average Distance: {np.mean(episode_distances):.3f}m")
    print(f"  Success Rate: {np.mean(episode_successes)*100:.1f}%")
    print("="*70)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train G1 Reaching with Stable-Baselines3 PPO')
    parser.add_argument('--scene', type=str,
                       default="../unitree_g1/g1_table_box_scene.xml",
                       help='Path to scene XML')
    parser.add_argument('--timesteps', type=int, default=100_000,
                       help='Total training timesteps (default: 100k)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Steps per update (default: 2048)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='PPO epochs (default: 10)')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda parameter (default: 0.95)')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments (default: 4, use 1 to disable)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--action-smoothing', type=float, default=0.3,
                       help='Action smoothing EMA coefficient (default: 0.3)')
    parser.add_argument('--smoothness-weight', type=float, default=1.0,
                       help='Weight for smoothness penalty (default: 1.0)')
    parser.add_argument('--action-scale', type=float, default=0.25,
                       help='Action scaling factor (default: 0.25)')
    parser.add_argument('--sim-substeps', type=int, default=10,
                       help='Physics substeps per RL step (default: 10)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable real-time plotting (useful for headless systems)')
    parser.add_argument('--test', type=str, default=None,
                       help='Test a trained model (provide path)')

    args = parser.parse_args()

    # Check scene file exists
    if not os.path.exists(args.scene):
        print(f"Error: Scene file not found: {args.scene}")
        exit(1)

    # Test mode
    if args.test:
        test_trained_model(args.test, args.scene)
    else:
        # Train mode
        model, env = train_g1_reaching(
            scene_path=args.scene,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gae_lambda=args.gae_lambda,
            seed=args.seed,
            device=args.device,
            action_smoothing=args.action_smoothing,
            smoothness_weight=args.smoothness_weight,
            action_scale=args.action_scale,
            sim_substeps=args.sim_substeps,
            n_envs=args.n_envs,
            enable_plot=not args.no_plot
        )

        print("\n[SUCCESS] Training pipeline complete!")
