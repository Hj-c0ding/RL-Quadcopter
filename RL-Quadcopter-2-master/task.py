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

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Extract state information
        position = self.sim.pose[:3]
        orientation = self.sim.pose[3:]  # phi, theta, psi (Euler angles)
        velocity = self.sim.v
        angular_velocity = self.sim.angular_v
        
        # Component 1: Position error (squared Euclidean distance)
        position_error = np.linalg.norm(position - self.target_pos)
        position_reward = -1.0 * (position_error ** 2)
        
        # Component 2: Velocity penalty (discourage high speeds for stability)
        velocity_penalty = -0.1 * np.linalg.norm(velocity) ** 2
        
        # Component 3: Orientation penalty (keep quadcopter level)
        orientation_penalty = -0.5 * np.sum(orientation ** 2)
        
        # Component 4: Angular velocity penalty (minimize rotation)
        angular_velocity_penalty = -0.05 * np.linalg.norm(angular_velocity) ** 2
        
        # Component 5: Success bonus (reaching target with low velocity)
        success_bonus = 0.0
        if position_error < 1.0 and np.linalg.norm(velocity) < 0.5:
            success_bonus = 10.0
        
        # Total reward
        reward = (position_reward + velocity_penalty + 
                  orientation_penalty + angular_velocity_penalty + 
                  success_bonus)
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        
        # Add crash penalty if episode ended prematurely (crashed)
        if done and self.sim.time < self.sim.runtime:
            reward -= 10.0  # Penalty for crashing before completing episode
        
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state