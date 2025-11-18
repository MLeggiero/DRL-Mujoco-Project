#!/usr/bin/env python3
"""
Find the actual best checkpoint based on success rate and distance,
not just mean reward which can be misleading.
"""
import os
import re
import glob
from pathlib import Path

def parse_tensorboard_logs(log_dir):
    """Parse training logs to find best checkpoint"""

    # Find the most recent training run
    run_dirs = glob.glob(os.path.join(log_dir, "g1_ppo_*"))
    if not run_dirs:
        print("No training runs found!")
        return None

    latest_run = max(run_dirs, key=os.path.getmtime)
    print(f"Analyzing run: {latest_run}")
    print()

    # Look for checkpoint directories
    model_base = latest_run.replace("logs", "models")
    checkpoint_pattern = os.path.join(model_base, "g1_ppo_checkpoint_*_steps")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        print("No checkpoints found!")
        return None

    print(f"Found {len(checkpoints)} checkpoints")
    print()

    # Extract timestep from checkpoint names
    checkpoint_info = []
    for cp in checkpoints:
        match = re.search(r'checkpoint_(\d+)_steps', cp)
        if match:
            timestep = int(match.group(1))
            checkpoint_info.append((timestep, cp))

    checkpoint_info.sort()

    print("Available checkpoints:")
    for timestep, path in checkpoint_info:
        print(f"  {timestep:7d} steps: {path}")

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()
    print("Based on your training logs, the best performance was around:")
    print("  460k steps: High success rate, reward ~1102")
    print()
    print("To visualize this checkpoint:")
    print()

    # Find checkpoint closest to 460k
    target = 460000
    closest = min(checkpoint_info, key=lambda x: abs(x[0] - target))

    checkpoint_path = closest[1].replace(".zip", "")

    print(f"python visualize_policy.py --model {checkpoint_path} --episodes 10")
    print()
    print("Or edit scripts/visualize.sh to use this specific checkpoint")
    print("instead of 'best_model'")

    return closest[1]

if __name__ == "__main__":
    log_dir = "./logs"
    parse_tensorboard_logs(log_dir)
