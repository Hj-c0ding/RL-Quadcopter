import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random



class ReplayBuffer:
    """Experience replay buffer for Q-learning"""
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class Agent(nn.Module):
    def __init__(self, task, hidden_size=128, learning_rate=0.0005, gamma=0.99):
        super(Agent, self).__init__()
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.gamma = gamma  # discount factor
        
        # Q-Network (estimates Q-values for actions)
        self.fc1 = nn.Linear(self.state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.action_size)
        
        # Target Q-Network (for stability)
        self.target_fc1 = nn.Linear(self.state_size, hidden_size)
        self.target_fc2 = nn.Linear(hidden_size, hidden_size)
        self.target_fc3 = nn.Linear(hidden_size, self.action_size)
        self._copy_weights()
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size=10000)
        self.batch_size = 64
        self.train_counter = 0
        self.train_interval = 4
        self.update_target_interval = 1000
        
        self.best_score = -np.inf
        
        self.reset_episode()
    
    def _copy_weights(self):
        """Copy main network weights to target network"""
        self.target_fc1.load_state_dict(self.fc1.state_dict())
        self.target_fc2.load_state_dict(self.fc2.state_dict())
        self.target_fc3.load_state_dict(self.fc3.state_dict())
    
    def forward(self, state):
        """Forward pass through Q-network"""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
    
    def _target_forward(self, state):
        """Forward pass through target Q-network"""
        x = torch.relu(self.target_fc1(state))
        x = torch.relu(self.target_fc2(x))
        q_values = self.target_fc3(x)
        return q_values
    
    def act(self, state):
        """Select action using Q-network"""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        action = q_values.cpu().numpy()[0]
        
        # Epsilon-greedy exploration
        if random.random() < 0.1:  # 10% separate noise/exploration chance
             action += np.random.normal(0, 50, size=self.action_size)

        # Clamp to valid rotor speeds
        action = np.clip(action, self.action_low, self.action_high)
        # Ensure minimum rotor speed to avoid division by zero
        action = np.maximum(action, 0.01)
        return action
    
    def reset_episode(self):
        """Reset episode statistics"""
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state
    
    def step(self, state, action, reward, next_state, done):
        """Store experience and train on batch"""
        self.total_reward += reward
        self.count += 1
        
        # Store experience
        self.memory.add(state, action, reward, next_state, done)
        
        # Train on batch if we have enough experiences
        if len(self.memory) > self.batch_size and self.train_counter % self.train_interval == 0:
            self.train_on_batch()
        
        self.train_counter += 1
        
        # Update target network periodically
        if self.train_counter % self.update_target_interval == 0:
            self._copy_weights()
        
        if done:
            self.learn()
    
    def train_on_batch(self):
        """Train Q-network on a batch of experiences"""
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q-values
        q_values = self.forward(states)
        
        # Target Q-values: Q_target = r + gamma * max(Q(s'))
        with torch.no_grad():
            next_q_values = self._target_forward(next_states)
            # For continuous action space: take max over action dimensions
            max_next_q = torch.amax(next_q_values, dim=1, keepdim=True)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute loss and backprop
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
    
    def learn(self):
        """Called at end of episode"""
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
