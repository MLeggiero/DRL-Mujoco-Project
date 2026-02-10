#!/usr/bin/env python3
"""
Monitor training progress from TensorBoard logs.
"""

import os
import sys
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import time


def monitor_training(log_dir, refresh_interval=10):
    """Monitor training from TensorBoard logs."""
    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"ERROR: Log directory not found: {log_path}")
        return

    # Find event file
    event_files = list(log_path.glob("**/events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_path}")
        return

    event_file = event_files[0]
    print(f"Monitoring: {event_file}")
    print("="*60)

    ea = event_accumulator.EventAccumulator(str(event_file.parent))
    ea.Reload()

    last_step = 0

    try:
        while True:
            ea.Reload()

            # Get available tags
            tags = ea.Tags()

            if 'scalars' in tags and len(tags['scalars']) > 0:
                print(f"\n[{time.strftime('%H:%M:%S')}] Training Progress:")
                print("-" * 60)

                # Print key metrics
                for tag in ['rollout/ep_rew_mean', 'rollout/success_rate',
                           'train/loss', 'time/fps']:
                    if tag in tags['scalars']:
                        events = ea.Scalars(tag)
                        if events:
                            latest = events[-1]
                            print(f"{tag:30s}: {latest.value:10.4f} (step {latest.step})")
                            last_step = max(last_step, latest.step)

                if last_step > 0:
                    print(f"\nLatest step: {last_step}")

            else:
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for training data...")

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "./training_test/tensorboard"
    monitor_training(log_dir)
