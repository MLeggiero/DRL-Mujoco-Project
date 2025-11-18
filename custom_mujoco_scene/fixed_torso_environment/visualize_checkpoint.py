#!/usr/bin/env python3
"""
Visualize a specific checkpoint by timestep number
"""
import argparse
import glob
import os
import re
from visualize_policy import visualize_policy

def find_checkpoint(timesteps):
    """Find checkpoint closest to specified timesteps"""
    # Find most recent model directory
    model_dirs = glob.glob("./models/g1_ppo_*")
    if not model_dirs:
        print("No model directories found!")
        return None

    latest_dir = max(model_dirs, key=os.path.getmtime)

    # Look for checkpoint
    checkpoint_pattern = os.path.join(latest_dir, f"g1_ppo_checkpoint_{timesteps}_steps.zip")
    if os.path.exists(checkpoint_pattern):
        return checkpoint_pattern.replace(".zip", "")

    # Try to find closest
    all_checkpoints = glob.glob(os.path.join(latest_dir, "g1_ppo_checkpoint_*_steps.zip"))
    checkpoint_info = []
    for cp in all_checkpoints:
        match = re.search(r'checkpoint_(\d+)_steps', cp)
        if match:
            ts = int(match.group(1))
            checkpoint_info.append((ts, cp))

    if not checkpoint_info:
        print("No checkpoints found!")
        return None

    # Find closest
    closest = min(checkpoint_info, key=lambda x: abs(x[0] - timesteps))
    print(f"Requested {timesteps}, using closest: {closest[0]}")

    return closest[1].replace(".zip", "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize specific checkpoint')
    parser.add_argument('--timesteps', type=int, default=460000,
                       help='Checkpoint timesteps (default: 460000 - best from your training)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to visualize')
    parser.add_argument('--fast', action='store_true',
                       help='Run fast (no slowdown)')

    args = parser.parse_args()

    checkpoint_path = find_checkpoint(args.timesteps)

    if checkpoint_path is None:
        exit(1)

    print("=" * 70)
    print(f"Visualizing checkpoint at {args.timesteps} steps")
    print("=" * 70)
    print()

    # Run visualization
    visualize_policy(
        model_path=checkpoint_path,
        num_episodes=args.episodes,
        slow=not args.fast
    )
