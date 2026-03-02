import numpy as np
from physics_sim import PhysicsSim


class TaskHover():

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        # Simulation — default start already at the hover target
        if init_pose is None:
            init_pose = np.array([0., 0., 10., 0., 0., 0.])
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 3

        # State: pose(6) + velocity(3) per action‑repeat
        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal — hover at (0, 0, 10)
        self.target_pos = (target_pos if target_pos is not None
                           else np.array([0., 0., 10.]))


    def get_reward(self):
    
        #  position error (Euclidean) 
        pos_error = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        # Gaussian‑style: 1 at target, decays with σ ≈ 3 m
        pos_reward = np.exp(-0.1 * pos_error ** 2)

        # z‑component gets extra weight (most important axis) 
        z_error = abs(self.sim.pose[2] - self.target_pos[2])
        z_reward = np.exp(-0.5 * z_error ** 2)

        #  velocity penalty 
        speed = np.linalg.norm(self.sim.v)
        vel_penalty = 0.05 * speed  # keep small so it doesn't dominate

        # orientation penalty (roll & pitch only) ---
        angle_penalty = 0.05 * (abs(self.sim.pose[3]) + abs(self.sim.pose[4]))

        # --- alive bonus ---
        alive = 0.1

        reward = pos_reward + z_reward - vel_penalty - angle_penalty + alive
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            # Append pose + velocity to state
            state_all.append(np.concatenate([self.sim.pose, self.sim.v]))
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state_single = np.concatenate([self.sim.pose, self.sim.v])
        state = np.concatenate([state_single] * self.action_repeat)
        return state
