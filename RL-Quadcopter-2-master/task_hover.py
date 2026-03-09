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
        """Maximize: stay at target z=10 and keep vertical velocity near zero."""
        z = self.sim.pose[2]
        z_error = abs(z - self.target_pos[2])
        vz = self.sim.v[2]
        # Altitude: strong reward at target, steep decay (max 2.0)
        z_reward = 2.0 * np.exp(-0.6 * z_error ** 2)
        # Vertical speed: strong reward for small |vz| (max 1.0)
        vz_reward = np.exp(-1.0 * vz ** 2)
        # Bonus for tight hover at target
        tight_bonus = 0.5 if z_error < 0.3 and abs(vz) < 0.2 else 0.0
        # Penalty for being above target (discourage floating high)
        above_penalty = 0.06 * max(0, z - self.target_pos[2])
        reward = z_reward + vz_reward + tight_bonus - 0.04 * z_error - 0.03 * abs(vz) - above_penalty
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
