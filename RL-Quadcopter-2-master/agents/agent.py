import numpy as np
import random
import torch
import torch.nn as nn


class Agent():
    def Quad_Copter_Agent(self, task, action_size=4, hidden_size=128):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
    def reset_episode(self):
        self.total_reward = 0.0
        state = self.task.reset()
        return state