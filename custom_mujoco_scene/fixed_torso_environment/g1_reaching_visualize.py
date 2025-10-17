"""
Visualization script for G1 Reaching Environment
Works with standard MuJoCo viewer
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from g1_reaching_env import G1ReachingEnv
from g1_reaching_training import RandomPolicy, load_policy

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_policy(policy, num_episodes=3, max_steps=500, slow_motion=False):
    """Visualize a policy in the MuJoCo viewer"""
    env = G1ReachingEnv(max_episode_steps=max_steps)
    
    print("Starting MuJoCo viewer...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom in/out")
    print("  - Double-click: Select body")
    print("  - Ctrl+Right-click: Move view")
    print("  - Backspace: Reset camera")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for episode in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            
            # Reset environment
            obs = env.reset(seed=episode)
            total_reward = 0
            
            for step in range(max_steps):
                # Get action from policy
                action = policy(obs)
                
                # Step environment
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                # Update viewer
                viewer.sync()
                
                # Print progress every 50 steps
                if (step + 1) % 50 == 0:
                    print(f"Step {step + 1}: "
                          f"Reward = {total_reward:.2f}, "
                          f"Distance = {info['distance']:.4f}m")
                
                # Slow motion for debugging
                if slow_motion:
                    time.sleep(0.01)
                
                # Check termination
                if done:
                    print(f"\nEpisode terminated at step {step + 1}")
                    if info['distance'] < 0.03:
                        print("✓ Target reached!")
                    elif env.data.xpos[env.pelvis_id][2] < 0.4:
                        print("✗ Robot fell")
                    break
            
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Final Distance: {info['distance']:.4f}m")
            
            # Pause between episodes
            if episode < num_episodes - 1:
                print("\nStarting next episode in 2 seconds...")
                time.sleep(2)


def visualize_random_policy(num_episodes=3, max_steps=500, slow_motion=False):
    """Visualize random policy"""
    print("Visualizing RANDOM policy")
    print("This is a baseline - expect poor performance")
    
    env = G1ReachingEnv(max_episode_steps=max_steps)
    policy = RandomPolicy(env.action_space_dim)
    
    visualize_policy(policy, num_episodes, max_steps, slow_motion)


def visualize_trained_policy(policy_file, num_episodes=3, max_steps=500, slow_motion=False):
    """Visualize trained policy"""
    print(f"Loading policy from {policy_file}...")
    policy = load_policy(policy_file)
    
    print("Visualizing TRAINED policy")
    visualize_policy(policy, num_episodes, max_steps, slow_motion)


def record_video(policy=None, policy_file=None, output_file='robot_reaching.mp4',
                max_steps=500, fps=30, width=1280, height=720):
    """Record a video of the policy"""
    try:
        import imageio
    except ImportError:
        print("Error: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return
    
    print(f"Recording video to {output_file}...")
    
    env = G1ReachingEnv(max_episode_steps=max_steps)
    
    # Load policy if file provided
    if policy_file:
        policy = load_policy(policy_file)
    elif policy is None:
        policy = RandomPolicy(env.action_space_dim)
    
    # Create renderer
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    
    # Reset environment
    obs = env.reset(seed=0)
    
    frames = []
    total_reward = 0
    
    print("Rendering frames...")
    for step in range(max_steps):
        # Get action
        action = policy(obs)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Render frame
        renderer.update_scene(env.data, camera="track")
        frame = renderer.render()
        frames.append(frame)
        
        # Print progress
        if (step + 1) % 100 == 0:
            print(f"  Rendered {step + 1}/{max_steps} frames...")
        
        # Check termination
        if done:
            print(f"Episode ended at step {step + 1}")
            break
    
    # Save video
    print(f"Saving video with {len(frames)} frames...")
    imageio.mimsave(output_file, frames, fps=fps)
    
    print(f"✓ Video saved to {output_file}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final Distance: {info['distance']:.4f}m")


def compare_policies(policy_files, num_episodes=3, max_steps=500):
    """Compare multiple trained policies side by side"""
    env = G1ReachingEnv(max_episode_steps=max_steps)
    
    policies = []
    for pfile in policy_files:
        print(f"Loading {pfile}...")
        policies.append(load_policy(pfile))
    
    print(f"\nComparing {len(policies)} policies over {num_episodes} episodes")
    print("="*60)
    
    results = [[] for _ in policies]
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        for i, policy in enumerate(policies):
            obs = env.reset(seed=episode)
            total_reward = 0
            
            for step in range(max_steps):
                action = policy(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            results[i].append({
                'reward': total_reward,
                'distance': info['distance'],
                'length': step + 1
            })
            
            print(f"  Policy {i+1}: Reward={total_reward:.2f}, "
                  f"Distance={info['distance']:.4f}m, Steps={step+1}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for i, policy_results in enumerate(results):
        avg_reward = np.mean([r['reward'] for r in policy_results])
        avg_distance = np.mean([r['distance'] for r in policy_results])
        avg_length = np.mean([r['length'] for r in policy_results])
        
        print(f"Policy {i+1} ({policy_files[i]}):")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Final Distance: {avg_distance:.4f}m")
        print(f"  Average Episode Length: {avg_length:.1f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize G1 reaching policy')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'trained', 'record', 'compare'],
                        help='Visualization mode')
    parser.add_argument('--policy', type=str, default=None,
                        help='Path to trained policy file (for trained/record mode)')
    parser.add_argument('--policies', nargs='+', default=None,
                        help='Multiple policy files for comparison')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to visualize')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--slow', action='store_true',
                        help='Enable slow motion')
    parser.add_argument('--output', type=str, default='robot_reaching.mp4',
                        help='Output video filename (for record mode)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS (for record mode)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Video width (for record mode)')
    parser.add_argument('--height', type=int, default=720,
                        help='Video height (for record mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'random':
        visualize_random_policy(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            slow_motion=args.slow
        )
    
    elif args.mode == 'trained':
        if not args.policy:
            print("Error: --policy argument required for trained mode")
            print("Example: python g1_reaching_visualize.py --mode trained --policy trained_policy.pkl")
        else:
            visualize_trained_policy(
                args.policy,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                slow_motion=args.slow
            )
    
    elif args.mode == 'record':
        record_video(
            policy_file=args.policy,
            output_file=args.output,
            max_steps=args.max_steps,
            fps=args.fps,
            width=args.width,
            height=args.height
        )
    
    elif args.mode == 'compare':
        if not args.policies or len(args.policies) < 2:
            print("Error: --policies argument with at least 2 files required for compare mode")
            print("Example: python g1_reaching_visualize.py --mode compare --policies policy1.pkl policy2.pkl")
        else:
            compare_policies(
                args.policies,
                num_episodes=args.episodes,
                max_steps=args.max_steps
            )