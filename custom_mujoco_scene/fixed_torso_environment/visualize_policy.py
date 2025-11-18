#!/usr/bin/env python3
"""
Visualize trained policy in MuJoCo viewer
Watch the robot execute the learned behavior in real-time
"""
import os
import sys
import argparse
import numpy as np
import time
import mujoco
import mujoco.viewer

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from g1_gym_wrapper import G1ReachingGymEnv


def visualize_policy(
    model_path,
    scene_path="../unitree_g1/g1_table_box_scene.xml",
    n_episodes=5,
    deterministic=True,
    slow_motion=False,
    show_target_marker=True
):
    """
    Visualize a trained policy in the MuJoCo viewer

    Args:
        model_path: Path to trained model (.zip file)
        scene_path: Path to MuJoCo scene XML
        n_episodes: Number of episodes to run
        deterministic: Use deterministic policy (True) or stochastic (False)
        slow_motion: Slow down visualization for better viewing
        show_target_marker: Highlight target object
    """

    print("="*70)
    print("POLICY VISUALIZATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print("="*70)
    print()

    # Load the model
    print("Loading model...")
    model = PPO.load(model_path)
    print("[OK] Model loaded")

    # Check for normalization stats
    vecnormalize_path = f"{model_path}_vecnormalize.pkl"
    has_normalization = os.path.exists(vecnormalize_path)

    # Create environment
    print("Creating environment...")
    env_fn = lambda: G1ReachingGymEnv(scene_path=scene_path)
    env = DummyVecEnv([env_fn])

    if has_normalization:
        print("Loading normalization statistics...")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
        print("[OK] Normalization loaded")

    print("[OK] Environment ready")
    print()

    # Get the underlying MuJoCo model for visualization
    base_env = env.envs[0].env

    # Launch MuJoCo viewer
    print("="*70)
    print("LAUNCHING MUJOCO VIEWER")
    print("="*70)
    print()
    print("Controls:")
    print("  - Mouse: Rotate camera (left drag), Pan (right drag), Zoom (scroll)")
    print("  - Double-click: Select body")
    print("  - Space: Pause/Resume")
    print("  - Esc: Close viewer")
    print()
    print("Press any key in the viewer to start...")
    print()

    # Statistics tracking
    all_rewards = []
    all_distances = []
    all_successes = []
    all_episode_lengths = []

    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        # Wait for user to press a key
        viewer.sync()

        for episode in range(n_episodes):
            print(f"\n{'='*70}")
            print(f"EPISODE {episode + 1}/{n_episodes}")
            print(f"{'='*70}")

            obs = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            min_distance = float('inf')

            # Episode start info
            info = base_env._get_observation()
            start_distance = info['distance_to_target']
            print(f"Starting distance: {start_distance:.3f}m")
            print(f"Target: {base_env.current_target}")
            print()

            while not done and step_count < 1000:
                # Get action from policy
                action, _states = model.predict(obs, deterministic=deterministic)

                # Step environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                step_count += 1

                # Track minimum distance
                current_distance = base_env._get_distance_to_target()
                min_distance = min(min_distance, current_distance)

                # Update viewer
                viewer.sync()

                # Slow motion if requested
                if slow_motion:
                    time.sleep(0.02)  # ~50 FPS

                # Print progress every 50 steps
                if step_count % 50 == 0:
                    print(f"  Step {step_count:3d} | "
                          f"Reward: {episode_reward:>7.2f} | "
                          f"Distance: {current_distance:.3f}m | "
                          f"Min: {min_distance:.3f}m")

            # Episode summary
            final_distance = base_env._get_distance_to_target()
            success = final_distance < 0.05

            all_rewards.append(episode_reward)
            all_distances.append(final_distance)
            all_successes.append(success)
            all_episode_lengths.append(step_count)

            print()
            print(f"Episode {episode + 1} Complete:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Final Distance: {final_distance:.3f}m")
            print(f"  Minimum Distance: {min_distance:.3f}m")
            print(f"  Success: {'YES' if success else 'NO'}")

            if episode < n_episodes - 1:
                print()
                print("Press any key in viewer to continue to next episode...")
                # Small pause between episodes
                time.sleep(1.0)

    # Overall statistics
    print()
    print("="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"Episodes: {n_episodes}")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average Distance: {np.mean(all_distances):.3f}m ± {np.std(all_distances):.3f}m")
    print(f"Average Length: {np.mean(all_episode_lengths):.1f} steps")
    print(f"Success Rate: {np.mean(all_successes)*100:.1f}% ({sum(all_successes)}/{n_episodes})")
    print(f"Best Distance: {min(all_distances):.3f}m")
    print("="*70)

    env.close()


def visualize_random_policy(
    scene_path="../unitree_g1/g1_table_box_scene.xml",
    n_episodes=3
):
    """Visualize random actions for comparison"""

    print("="*70)
    print("RANDOM POLICY VISUALIZATION (Baseline)")
    print("="*70)
    print()

    from g1_rl_environment import G1ReachTouchEnv
    env = G1ReachTouchEnv(scene_path=scene_path)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}/{n_episodes} (Random Actions)")

            obs = env.reset()
            episode_reward = 0

            for step in range(300):
                # Random action
                action = np.random.uniform(-1, 1, env.n_actions)

                obs, reward, done, info = env.step(action)
                episode_reward += reward

                viewer.sync()
                time.sleep(0.01)

                if step % 50 == 0:
                    distance = obs['distance_to_target']
                    print(f"  Step {step:3d} | Reward: {episode_reward:>7.2f} | Distance: {distance:.3f}m")

                if done:
                    break

            final_distance = obs['distance_to_target']
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Final Distance={final_distance:.3f}m")

            time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize trained G1 reaching policy')
    parser.add_argument('--model', type=str,
                       help='Path to trained model (e.g., ./models/g1_ppo_*/best_model/best_model)')
    parser.add_argument('--scene', type=str,
                       default="../unitree_g1/g1_table_box_scene.xml",
                       help='Path to scene XML')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to visualize')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (default: deterministic)')
    parser.add_argument('--slow', action='store_true',
                       help='Slow motion for better viewing')
    parser.add_argument('--random', action='store_true',
                       help='Visualize random policy instead of trained')
    parser.add_argument('--list-models', action='store_true',
                       help='List available trained models')

    args = parser.parse_args()

    # List available models
    if args.list_models:
        print("="*70)
        print("AVAILABLE TRAINED MODELS")
        print("="*70)

        models_dir = "./models"
        if os.path.exists(models_dir):
            for run_dir in sorted(os.listdir(models_dir)):
                run_path = os.path.join(models_dir, run_dir)
                if os.path.isdir(run_path):
                    print(f"\n{run_dir}:")

                    # Check for best model
                    best_model = os.path.join(run_path, "best_model", "best_model.zip")
                    if os.path.exists(best_model):
                        print(f"  [BEST]  {best_model}")

                    # Check for final model
                    final_model = os.path.join(run_path, "final_model.zip")
                    if os.path.exists(final_model):
                        print(f"  [FINAL] {final_model}")

                    # Check for checkpoints
                    for item in os.listdir(run_path):
                        if item.endswith(".zip") and "checkpoint" in item:
                            checkpoint_path = os.path.join(run_path, item)
                            print(f"  [CKPT]  {checkpoint_path}")
        else:
            print("No models directory found. Train a model first:")
            print("  python train_sb3.py")

        print()
        print("To visualize a model:")
        print("  python visualize_policy.py --model <path_to_model>")
        sys.exit(0)

    # Check scene file
    if not os.path.exists(args.scene):
        print(f"Error: Scene file not found: {args.scene}")
        sys.exit(1)

    # Visualize random or trained policy
    if args.random:
        visualize_random_policy(args.scene, args.episodes)
    else:
        if not args.model:
            print("Error: --model required (or use --random for random policy)")
            print("\nUse --list-models to see available models")
            print("Example: python visualize_policy.py --model ./models/g1_ppo_*/best_model/best_model")
            sys.exit(1)

        if not os.path.exists(args.model + ".zip"):
            print(f"Error: Model not found: {args.model}.zip")
            print("\nUse --list-models to see available models")
            sys.exit(1)

        visualize_policy(
            model_path=args.model,
            scene_path=args.scene,
            n_episodes=args.episodes,
            deterministic=not args.stochastic,
            slow_motion=args.slow
        )
