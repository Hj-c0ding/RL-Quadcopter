import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple


class Actor(nn.Module):
    """Actor (Policy) Model for DDPG."""
    
    def __init__(self, state_size, action_size, action_low, action_high, hidden_size=256):
        super(Actor, self).__init__()
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = action_high - action_low
        
        # Network architecture
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights with small values."""
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        """Map state to action values."""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        # Sigmoid to bound output to [0, 1], then scale to action range
        x = torch.sigmoid(self.fc3(x))
        return x * self.action_range + self.action_low


class Critic(nn.Module):
    """Critic (Value) Model for DDPG."""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights with small values."""
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        """Map (state, action) pairs to Q-values."""
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration."""
    
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        """Reset the internal state to mean (mu)."""
        self.state = self.mu.copy()
    
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.FloatTensor(np.vstack([e.state for e in experiences if e is not None]))
        actions = torch.FloatTensor(np.vstack([e.action for e in experiences if e is not None]))
        rewards = torch.FloatTensor(np.vstack([e.reward for e in experiences if e is not None]))
        next_states = torch.FloatTensor(np.vstack([e.next_state for e in experiences if e is not None]))
        dones = torch.FloatTensor(np.vstack([e.done for e in experiences if e is not None]))
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent():
    """DDPG Agent for continuous control."""
    
    def __init__(self, task, hidden_size=256, buffer_size=100000, batch_size=64, 
                 gamma=0.99, tau=0.001, lr_actor=1e-4, lr_critic=1e-3):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        # Hyperparameters
        self.gamma = gamma  # discount factor
        self.tau = tau      # soft update of target parameters
        self.batch_size = batch_size
        
        # Actor Networks (local and target)
        self.actor_local = Actor(self.state_size, self.action_size, 
                                  self.action_low, self.action_high, hidden_size)
        self.actor_target = Actor(self.state_size, self.action_size, 
                                   self.action_low, self.action_high, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Networks (local and target)
        self.critic_local = Critic(self.state_size, self.action_size, hidden_size)
        self.critic_target = Critic(self.state_size, self.action_size, hidden_size)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        
        # Initialize target networks to same weights
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        
        # Noise process for exploration
        self.noise = OUNoise(self.action_size, mu=(self.action_low + self.action_high) / 2)
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
        # Episode tracking
        self.reset_episode()
        
        # Score tracking for compatibility with existing code
        self.best_score = -np.inf
        self.score = 0.0
    
    def reset_episode(self):
        """Reset for a new episode."""
        self.noise.reset()
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        self.last_state = state
        return state
    
    def act(self, state, add_noise=True):
        """Return actions for given state as per current policy."""
        state = torch.FloatTensor(state.reshape(1, -1))
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy()
        self.actor_local.train()
        
        # Add noise for exploration
        if add_noise:
            action += self.noise.sample()
        
        # Clip to valid range
        return np.clip(action[0], self.action_low, self.action_high)
    
    def step(self, reward, done):
        """Save experience and trigger learning (simplified for compatibility)."""
        self.total_reward += reward
        self.count += 1
        
        # Update score when episode is done
        if done:
            self.score = self.total_reward / float(self.count) if self.count else 0.0
            if self.score > self.best_score:
                self.best_score = self.score
    
    def step_with_next_state(self, state, action, reward, next_state, done):
        """Complete step method that includes experience replay and learning."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.total_reward += reward
        self.count += 1
        
        # Learn if enough samples are available
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        
        # Update score when episode is done
        if done:
            self.score = self.total_reward / float(self.count) if self.count else 0.0
            if self.score > self.best_score:
                self.best_score = self.score
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # Update Critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # Update Actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)