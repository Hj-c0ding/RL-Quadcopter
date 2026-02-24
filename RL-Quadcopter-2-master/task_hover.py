import numpy as np
from physics_sim import PhysicsSim

class TaskHover():
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
        # Reward for hovering:
        # 1. Penalize distance from target position (x, y, z)
        # 2. Penalize velocity (we want it to stay still)
        # 3. Penalize extreme angles (we want it to stay flat)
        
        # Distance from target
        dist_penalty = 0.3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # Velocity penalty (we want to hover, so velocity should be low)
        # However, we might want to prioritize position first. 
        # Let's keep it simple for now, similar to original but maybe tweaked.
        
        # A common reward for hover is closer to 1 if close to target, and falls off.
        # Original: reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # Let's try to improve stability by penalizing high velocities slightly?
        # For now, I will stick to a position-based reward but ensure start/end matching makes sense.
        # The user said "start in the air and to hover".
        
        reward = 1. - .3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # Optional: Add bonus for staying alive (not crashing) implicitly handled by runtime if episode ends on crash?
        # PhysicsSim ends if z < 0.
        
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
