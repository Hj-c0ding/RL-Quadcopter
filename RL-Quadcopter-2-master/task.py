import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal: Default to hovering at altitude 10m (starting from ground level z=0)
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward for takeoff and hover task.
        
        Reward components:
        - Base reward for not crashing
        - Penalty for distance from target position (x, y, z)
        - Penalty for high velocity (want stable, stationary hovering)
        - Penalty for large pitch/roll angles (want level flight)
        
        The reward is carefully designed to avoid divergence while encouraging progress.
        """
        # Base reward for not crashing/flying too far
        reward = 1.0
        
        # Position penalty: penalize distance from target position
        # Use sqrt to compress large distances, preventing divergence
        position_error = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        # Cap the position error to prevent extreme penalties
        position_error = min(position_error, 50.0)  # Max 50m penalty
        reward -= 0.3 * position_error
        
        # Velocity penalty: penalize high velocity when at target
        # Encourages the agent to decelerate and hover stably
        velocity_norm = np.linalg.norm(self.sim.v)
        velocity_norm = min(velocity_norm, 30.0)  # Cap at 30 m/s
        reward -= 0.1 * velocity_norm
        
        # Angle penalty: penalize large pitch and roll angles
        # Keep yaw unrestricted (it's less critical for hovering)
        # phi is roll (index 3), theta is pitch (index 4)
        angle_penalty = np.abs(self.sim.pose[3]) + np.abs(self.sim.pose[4])
        angle_penalty = min(angle_penalty, 6.28)  # Cap at ±180 degrees
        reward -= 0.15 * angle_penalty
        
        # Ensure reward is finite and not extreme
        reward = np.nan_to_num(reward, nan=-100.0, posinf=-100.0, neginf=-100.0)
        reward = np.clip(reward, -100.0, 1.0)  # Bound range
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state