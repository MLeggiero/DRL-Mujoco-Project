#!/usr/bin/env python3
"""
Motion planning for VLM-based grasping.

Provides:
- Inverse kinematics solver
- Trajectory generation
- Collision checking
- Grasp execution
"""

import numpy as np
import mujoco
from typing import Tuple, List, Optional
import warnings


class IKSolver:
    """Inverse kinematics solver using MuJoCo's Jacobian."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize IK solver.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

    def solve(self,
             target_pos: np.ndarray,
             target_quat: Optional[np.ndarray] = None,
             site_name: str = 'right_palm',
             max_iterations: int = 100,
             tolerance: float = 0.001,
             step_size: float = 0.1,
             regularization: float = 0.01) -> Tuple[np.ndarray, bool]:
        """
        Solve IK for target end-effector pose.

        Args:
            target_pos: (3,) target position
            target_quat: Optional (4,) target quaternion [w,x,y,z]
            site_name: Name of end-effector site
            max_iterations: Maximum IK iterations
            tolerance: Position error tolerance (meters)
            step_size: IK step size (0-1)
            regularization: Damping factor for stability

        Returns:
            Tuple of (joint_positions, success)
        """
        # Get site ID
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site '{site_name}' not found")

        # Save initial state
        initial_qpos = self.data.qpos.copy()

        # IK loop
        for iteration in range(max_iterations):
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Get current end-effector position
            current_pos = self.data.site(site_name).xpos.copy()

            # Position error
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)

            # Check convergence
            if error_norm < tolerance:
                return self.data.qpos.copy(), True

            # Compute Jacobian (position only for now)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

            # Damped least squares IK
            # Δq = J^T (J J^T + λI)^{-1} e
            J = jacp
            lambda_mat = regularization * np.eye(3)

            try:
                dq = J.T @ np.linalg.solve(J @ J.T + lambda_mat, pos_error)
            except np.linalg.LinAlgError:
                warnings.warn("IK singular matrix, restoring initial pose")
                self.data.qpos[:] = initial_qpos
                return initial_qpos, False

            # Update joint positions
            self.data.qpos[:] += step_size * dq

            # Clamp to joint limits
            self.data.qpos[:] = np.clip(
                self.data.qpos,
                self.model.jnt_range[:, 0],
                self.model.jnt_range[:, 1]
            )

        # Failed to converge
        warnings.warn(f"IK failed to converge after {max_iterations} iterations (error={error_norm:.4f}m)")
        self.data.qpos[:] = initial_qpos
        return initial_qpos, False

    def solve_with_orientation(self,
                              target_pos: np.ndarray,
                              target_quat: np.ndarray,
                              site_name: str = 'right_palm',
                              **kwargs) -> Tuple[np.ndarray, bool]:
        """
        Solve IK with position and orientation constraints.

        Args:
            target_pos: (3,) target position
            target_quat: (4,) target quaternion [w,x,y,z]
            site_name: End-effector site name
            **kwargs: Additional arguments for solve()

        Returns:
            Tuple of (joint_positions, success)
        """
        # TODO: Implement full 6-DOF IK with orientation
        # For now, just use position-only IK
        return self.solve(target_pos, site_name=site_name, **kwargs)


class TrajectoryPlanner:
    """Plans smooth trajectories between configurations."""

    @staticmethod
    def interpolate_linear(start: np.ndarray,
                          goal: np.ndarray,
                          num_steps: int = 50) -> np.ndarray:
        """
        Linear interpolation between configurations.

        Args:
            start: (n,) start configuration
            goal: (n,) goal configuration
            num_steps: Number of waypoints

        Returns:
            (num_steps, n) trajectory
        """
        alphas = np.linspace(0, 1, num_steps)
        trajectory = np.outer(1 - alphas, start) + np.outer(alphas, goal)
        return trajectory

    @staticmethod
    def interpolate_smooth(start: np.ndarray,
                          goal: np.ndarray,
                          num_steps: int = 50) -> np.ndarray:
        """
        Smooth interpolation using minimum jerk trajectory.

        Args:
            start: (n,) start configuration
            goal: (n,) goal configuration
            num_steps: Number of waypoints

        Returns:
            (num_steps, n) trajectory
        """
        t = np.linspace(0, 1, num_steps)

        # Minimum jerk polynomial: 10t^3 - 15t^4 + 6t^5
        s = 10 * t**3 - 15 * t**4 + 6 * t**5

        trajectory = np.outer(1 - s, start) + np.outer(s, goal)
        return trajectory

    @staticmethod
    def compute_waypoint_times(trajectory: np.ndarray,
                               max_velocity: float = 1.0,
                               max_acceleration: float = 2.0) -> np.ndarray:
        """
        Compute timing for waypoints based on velocity/acceleration limits.

        Args:
            trajectory: (n_steps, n_dof) trajectory
            max_velocity: Maximum joint velocity (rad/s)
            max_acceleration: Maximum joint acceleration (rad/s^2)

        Returns:
            (n_steps,) array of times for each waypoint
        """
        n_steps = len(trajectory)
        times = np.zeros(n_steps)

        for i in range(1, n_steps):
            # Distance between waypoints
            delta = trajectory[i] - trajectory[i-1]
            distance = np.linalg.norm(delta)

            # Time needed based on velocity limit
            time_vel = distance / max_velocity

            # Time needed based on acceleration limit (approximate)
            time_acc = 2 * np.sqrt(distance / max_acceleration)

            # Take the maximum
            dt = max(time_vel, time_acc)
            times[i] = times[i-1] + dt

        return times


class MotionPlanner:
    """High-level motion planner for grasping."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize motion planner.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        self.ik_solver = IKSolver(model, data)
        self.trajectory_planner = TrajectoryPlanner()

    def plan_to_grasp(self,
                     grasp_pos: np.ndarray,
                     grasp_quat: Optional[np.ndarray] = None,
                     approach_distance: float = 0.15,
                     approach_steps: int = 50,
                     grasp_steps: int = 30) -> Tuple[np.ndarray, bool]:
        """
        Plan a grasp motion with approach phase.

        Args:
            grasp_pos: (3,) grasp position
            grasp_quat: Optional (4,) grasp orientation
            approach_distance: Distance to pre-grasp position (meters)
            approach_steps: Number of steps for approach motion
            grasp_steps: Number of steps for grasp motion

        Returns:
            Tuple of (trajectory, success)
            - trajectory: (total_steps, n_joints) waypoints
            - success: Whether IK succeeded
        """
        # Current configuration
        current_joints = self.data.qpos.copy()

        # Pre-grasp position (approach from above)
        pre_grasp_pos = grasp_pos.copy()
        pre_grasp_pos[2] += approach_distance  # Move up

        # Solve IK for pre-grasp
        pre_grasp_joints, success1 = self.ik_solver.solve(
            pre_grasp_pos,
            target_quat=grasp_quat,
            site_name='right_palm'
        )

        if not success1:
            warnings.warn("Failed to solve IK for pre-grasp pose")
            return np.array([current_joints]), False

        # Solve IK for grasp
        grasp_joints, success2 = self.ik_solver.solve(
            grasp_pos,
            target_quat=grasp_quat,
            site_name='right_palm'
        )

        if not success2:
            warnings.warn("Failed to solve IK for grasp pose")
            return np.array([current_joints]), False

        # Generate trajectory
        # Phase 1: Current -> Pre-grasp
        traj_to_pre = self.trajectory_planner.interpolate_smooth(
            current_joints,
            pre_grasp_joints,
            num_steps=approach_steps
        )

        # Phase 2: Pre-grasp -> Grasp (slower, more careful)
        traj_to_grasp = self.trajectory_planner.interpolate_smooth(
            pre_grasp_joints,
            grasp_joints,
            num_steps=grasp_steps
        )

        # Concatenate
        full_trajectory = np.vstack([traj_to_pre, traj_to_grasp])

        return full_trajectory, True

    def check_collision(self, qpos: np.ndarray) -> bool:
        """
        Check if configuration is in collision.

        Args:
            qpos: Joint configuration

        Returns:
            True if in collision, False otherwise
        """
        # Set configuration
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # Check for contacts (excluding floor and table)
        in_collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get geom names
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            # Skip floor and table contacts
            if geom1_name in ['floor', 'table_top'] or geom2_name in ['floor', 'table_top']:
                continue

            # Check for self-collision or collision with obstacles
            # (customize this based on your scene)
            in_collision = True
            break

        # Restore configuration
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)

        return in_collision

    def filter_collision_free(self, trajectory: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Filter trajectory to remove collision waypoints.

        Args:
            trajectory: (n_steps, n_joints) trajectory

        Returns:
            Tuple of (filtered_trajectory, is_valid)
        """
        valid_waypoints = []

        for waypoint in trajectory:
            if not self.check_collision(waypoint):
                valid_waypoints.append(waypoint)
            else:
                # Collision detected
                warnings.warn("Collision detected in trajectory")
                break

        if len(valid_waypoints) == len(trajectory):
            return trajectory, True
        elif len(valid_waypoints) > 0:
            return np.array(valid_waypoints), False
        else:
            return trajectory[:1], False  # Return at least current pose


# Example usage
if __name__ == "__main__":
    print("Motion planner module loaded successfully")
    print("\nAvailable classes:")
    print("  - IKSolver: Inverse kinematics using Jacobian")
    print("  - TrajectoryPlanner: Smooth trajectory generation")
    print("  - MotionPlanner: High-level grasp motion planning")
    print("\nFeatures:")
    print("  ✓ Damped least-squares IK")
    print("  ✓ Minimum jerk trajectories")
    print("  ✓ Collision checking")
    print("  ✓ Pre-grasp approaching")
