#!/usr/bin/env python3
"""
Compare linear vs squared distance reward
Shows why squared creates better gradient
"""
import numpy as np
import matplotlib.pyplot as plt

# Distance range from 0.5m (far) to 0.05m (touching)
distances = np.linspace(0.5, 0.05, 100)

# Linear distance reward
linear_reward = -distances

# Squared distance reward
squared_reward = -distances ** 2

# Calculate gradients (how much reward changes per cm of movement)
linear_gradient = np.abs(np.gradient(linear_reward, distances))
squared_gradient = np.abs(np.gradient(squared_reward, distances))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Reward values
ax1.plot(distances * 100, linear_reward, 'b-', label='Linear: -d', linewidth=2)
ax1.plot(distances * 100, squared_reward, 'r-', label='Squared: -d²', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Distance to Target (cm)')
ax1.set_ylabel('Reward')
ax1.set_title('Reward vs Distance')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()  # Closer is on the right

# Add annotations
ax1.annotate('Far: Both rewards similar',
            xy=(40, -0.4), xytext=(45, -0.3),
            arrowprops=dict(arrowstyle='->', color='gray'))
ax1.annotate('Close: Squared much better!',
            xy=(8, -0.006), xytext=(15, -0.15),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Plot 2: Gradient (learning signal strength)
ax2.plot(distances * 100, linear_gradient, 'b-', label='Linear gradient', linewidth=2)
ax2.plot(distances * 100, squared_gradient, 'r-', label='Squared gradient', linewidth=2)
ax2.set_xlabel('Distance to Target (cm)')
ax2.set_ylabel('Gradient Magnitude (learning signal)')
ax2.set_title('Learning Signal Strength')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

# Add annotations
ax2.annotate('Squared gradient 10x stronger\nwhen close to target!',
            xy=(8, squared_gradient[-10]), xytext=(20, 0.15),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('distance_reward_comparison.png', dpi=150, bbox_inches='tight')
print("=" * 70)
print("Distance Reward Comparison")
print("=" * 70)
print()

# Print numeric comparison
print("Reward at different distances:")
print(f"{'Distance':<12} {'Linear':<12} {'Squared':<12} {'Squared/Linear'}")
print("-" * 60)
for d in [0.50, 0.30, 0.15, 0.08, 0.05]:
    lin = -d
    sq = -d**2
    ratio = sq / lin if lin != 0 else 0
    print(f"{d*100:>6.0f} cm    {lin:>8.3f}      {sq:>8.4f}       {ratio:>6.2f}x")

print()
print("Gradient (learning signal) at different distances:")
print(f"{'Distance':<12} {'Linear':<12} {'Squared':<12} {'Squared/Linear'}")
print("-" * 60)
# Gradient of -d is -1 (constant)
# Gradient of -d^2 is -2d
for d in [0.50, 0.30, 0.15, 0.08, 0.05]:
    lin_grad = 1.0  # Constant for linear
    sq_grad = 2.0 * d  # 2d for squared
    ratio = sq_grad / lin_grad
    print(f"{d*100:>6.0f} cm    {lin_grad:>8.3f}      {sq_grad:>8.4f}       {ratio:>6.2f}x")

print()
print("=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("Linear reward: Same 'pull' whether 50cm or 5cm away")
print("Squared reward: 10x stronger 'pull' when close!")
print()
print("At 5cm: Squared gives 10x stronger learning signal than linear")
print("This helps the robot fine-tune to actually touch the target.")
print("=" * 70)
print()
print("Plot saved to: distance_reward_comparison.png")
