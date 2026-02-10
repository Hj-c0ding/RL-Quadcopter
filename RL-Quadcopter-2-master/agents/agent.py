import numpy as np
import random
import torch
import torch.nn as nn


class Agent(nn.Module):
    def Quad_Copter_Agent(self, task, action_size=4, hidden_size=128):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high


        #nueral network with 2 hidden layers
        self.fc1 = nn.Linear(self.state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #outputs the q value for each action, based on how well the action will do
        self.fc3 = nn.Linear(hidden_size, self.action_size)
    
    def act(self):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]

    def reset_episode(self):
        self.total_reward = 0.0
        state = self.task.reset()
        return state