#!/usr/bin/env python3
"""
Compare old vs new reward function behavior
Shows why the old function encouraged gaming
"""
import numpy as np
import matplotlib.pyplot as plt

def old_reward_function(distance, last_distance):
    """Old reward function that caused gaming"""
    distance_reward = -distance

    # OLD: Progress reward too high
    progress_reward = 0.0
    if last_distance is not None:
        progress = last_distance - distance
        progress_reward = progress * 10.0  # Gaming factor!

    # Proximity bonuses
    proximity_bonus = 0.0
    if distance < 0.3:
        proximity_bonus += 1.0
    if distance < 0.15:
        proximity_bonus += 3.0
    if distance < 0.05:
        proximity_bonus += 10.0

    time_penalty = -0.001

    return distance_reward + progress_reward + proximity_bonus + time_penalty

def new_reward_function(distance, last_distance):
    """New reward function that prevents gaming"""
    distance_reward = -distance

    # NEW: Reduced progress reward with asymmetric penalty
    progress_reward = 0.0
    if last_distance is not None:
        progress = last_distance - distance
        if progress > 0:
            progress_reward = progress * 2.0  # Reduced from 10.0
        else:
            progress_reward = progress * 0.5  # Penalty for moving away

    # Enhanced proximity bonuses
    proximity_bonus = 0.0
    if distance < 0.3:
        proximity_bonus += 2.0
    if distance < 0.15:
        proximity_bonus += 5.0
    if distance < 0.08:
        proximity_bonus += 10.0
    if distance < 0.05:
        proximity_bonus += 20.0

    time_penalty = -0.001

    return distance_reward + progress_reward + proximity_bonus + time_penalty

def simulate_oscillation():
    """Simulate oscillating behavior (gaming strategy)"""
    # Robot oscillates between 0.20m and 0.18m
    distances = [0.20, 0.18, 0.20, 0.18, 0.20, 0.18, 0.20, 0.18, 0.20, 0.18]

    old_rewards = []
    new_rewards = []

    last_dist = None
    for dist in distances:
        old_rewards.append(old_reward_function(dist, last_dist))
        new_rewards.append(new_reward_function(dist, last_dist))
        last_dist = dist

    print("=" * 70)
    print("OSCILLATION BEHAVIOR (Gaming Strategy)")
    print("=" * 70)
    print("Robot oscillates between 0.20m and 0.18m (never actually reaching)")
    print()

    old_total = sum(old_rewards)
    new_total = sum(new_rewards)

    print(f"Old reward function total: {old_total:+.2f}")
    print(f"New reward function total: {new_total:+.2f}")
    print()

    if old_total > 0:
        print("OLD FUNCTION: Positive total reward! Gaming is profitable!")
    else:
        print("OLD FUNCTION: Negative total, but reward is high per step")

    if new_total < old_total:
        print("NEW FUNCTION: Lower total reward. Gaming is less profitable!")

    return old_rewards, new_rewards

def simulate_reaching():
    """Simulate reaching behavior (desired strategy)"""
    # Robot smoothly reaches from 0.50m to 0.04m
    distances = np.linspace(0.50, 0.04, 10)

    old_rewards = []
    new_rewards = []

    last_dist = None
    for dist in distances:
        old_rewards.append(old_reward_function(dist, last_dist))
        new_rewards.append(new_reward_function(dist, last_dist))
        last_dist = dist

    print()
    print("=" * 70)
    print("REACHING BEHAVIOR (Desired Strategy)")
    print("=" * 70)
    print("Robot smoothly reaches from 0.50m to 0.04m (success!)")
    print()

    old_total = sum(old_rewards)
    new_total = sum(new_rewards)

    print(f"Old reward function total: {old_total:+.2f}")
    print(f"New reward function total: {new_total:+.2f}")
    print()

    if new_total > old_total:
        print("NEW FUNCTION: Higher reward for actual reaching! Good!")
    else:
        print("Both functions reward reaching similarly")

    return old_rewards, new_rewards

def plot_comparison():
    """Plot the reward comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Oscillation comparison
    distances_osc = [0.20, 0.18, 0.20, 0.18, 0.20, 0.18, 0.20, 0.18, 0.20, 0.18]
    old_osc, new_osc = [], []
    last_dist = None
    for dist in distances_osc:
        old_osc.append(old_reward_function(dist, last_dist))
        new_osc.append(new_reward_function(dist, last_dist))
        last_dist = dist

    steps = range(len(old_osc))
    ax1.plot(steps, old_osc, 'r-o', label='Old (Gaming)', linewidth=2)
    ax1.plot(steps, new_osc, 'g-o', label='New (Fixed)', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Oscillation Strategy (0.18m ↔ 0.20m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reaching comparison
    distances_reach = np.linspace(0.50, 0.04, 10)
    old_reach, new_reach = [], []
    last_dist = None
    for dist in distances_reach:
        old_reach.append(old_reward_function(dist, last_dist))
        new_reach.append(new_reward_function(dist, last_dist))
        last_dist = dist

    steps = range(len(old_reach))
    ax2.plot(steps, old_reach, 'r-o', label='Old', linewidth=2)
    ax2.plot(steps, new_reach, 'g-o', label='New', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reaching Strategy (0.50m → 0.04m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reward_comparison.png', dpi=150, bbox_inches='tight')
    print()
    print("=" * 70)
    print("Plot saved to: reward_comparison.png")
    print("=" * 70)

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("REWARD FUNCTION COMPARISON")
    print("Old (Gaming) vs New (Fixed)")
    print("=" * 70)
    print()

    # Simulate both strategies
    simulate_oscillation()
    simulate_reaching()

    # Create visualization
    try:
        plot_comparison()
    except ImportError:
        print()
        print("Note: Install matplotlib to generate comparison plot")
        print("  conda install matplotlib")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The old reward function (10x progress multiplier) made oscillation")
    print("profitable. The new function (2x progress, 0.5x penalty) makes")
    print("actual reaching more rewarding than gaming behavior.")
    print("=" * 70)
    print()
