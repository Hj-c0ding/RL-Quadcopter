import numpy as np
from collections import deque
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    """Actor outputs residual action in [-1, 1]; base hover thrust is added in Agent.act()."""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))  # [-1, 1] = residual
        return x


class Critic(nn.Module):
    """Critic (Value) Network that maps (state, action) pairs to Q-values."""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        """Output Q-value for state-action pair."""
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Experience replay buffer for DDPG."""
    
    def __init__(self, buffer_size=100000, batch_size=64):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        """Sample random batch from replay buffer."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)


class Agent:
    """DDPG Agent for continuous control. Uses residual policy: action = hover_thrust + delta."""
    
    # Rotor speed per motor for hover (slightly above physics equilibrium for margin)
    HOVER_ROTOR_SPEED = 404
    
    def __init__(self, state_size, action_size, action_low=0, action_high=900, seed=42, target_z=10.0):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.hover_action = np.ones(4, dtype=np.float64) * self.HOVER_ROTOR_SPEED
        self.delta_scale = 10.0  # policy can add ±130 thrust for corrections
        self.vz_gain = 2.0  # P-term: add thrust when falling (action += -vz_gain * vz)
        self.z_gain = 10.0  # P-term: add thrust to reduce z error
        self.target_z = target_z
        self._hover_tensor = None  # set on first use in learn() to match device
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor networks (current and target)
        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.00005)
        
        # Critic networks (current and target)
        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        # Initialize target networks with weights from current networks
        self._hard_update(self.actor, self.actor_target)
        self._hard_update(self.critic, self.critic_target)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=64)
        
        # Exploration parameters (no per-episode reset: decay over full training)
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.997
        
        # DDPG hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001  # Soft update factor
        self.min_batch_size = 64
        
        # Default path for best actor checkpoint (use consistent policy at test time)
        self.best_actor_path = "best_actor.pth"
    
    def save_actor(self, path=None):
        """Save current actor state dict for best-model checkpoint."""
        p = path or self.best_actor_path
        torch.save(self.actor.state_dict(), p)
    
    def load_actor(self, path=None):
        """Load actor from checkpoint (e.g. best model from training)."""
        import os
        p = path or self.best_actor_path
        if os.path.isfile(p):
            self.actor.load_state_dict(torch.load(p, map_location=self.device))
            self._hard_update(self.actor, self.actor_target)
        
    def _hard_update(self, local_model, target_model):
        """Hard update: copy weights from local model to target model."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
            
    def _soft_update(self, local_model, target_model):
        """Soft update: gradually blend weights from local to target model."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def _state_layout(self, state_size):
        """Infer state layout to pick z and vz indices."""
        if state_size % 9 == 0:
            return -7, -1  # pose(6)+v(3) per repeat
        if state_size % 6 == 0:
            return -4, None  # pose(6) per repeat
        return 2 if state_size > 2 else -1, None
    
    def act(self, state, training=False):
        """
        Residual policy: action = hover_thrust + delta + P(vz).
        P-term opposes vertical velocity (add thrust when falling).
        """
        state = np.asarray(state, dtype=np.float64)
        z_index, vz_index = self._state_layout(state.size)
        vz = state[vz_index] if vz_index is not None else 0.0
        state_t = torch.from_numpy(state).float().to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            residual = self.actor(state_t.unsqueeze(0)).squeeze(0).cpu().numpy()  # [-1, 1]
        self.actor.train()
        
        delta = residual * self.delta_scale
        action = self.hover_action + delta

        deltaz = self.target_z - state[z_index]
        # Proportional correction: when falling (vz < 0), add thrust on all rotors
        action += np.ones(4, dtype=np.float64) * (self.vz_gain * (-vz))
        action += np.ones(4, dtype=np.float64) * (self.z_gain * deltaz)

        if training:
            noise = np.random.normal(0, self.epsilon * 25, size=self.action_size)
            action = action + noise
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return np.clip(action, self.action_low, self.action_high)
    
    def step(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer and learn if enough samples are available.
        
        Args:
            state: Current state
            action: Action taken (scaled to [action_low, action_high])
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Scale action from [action_low, action_high] back to [-1, 1] for storage
        action_scaled = (action - self.action_low) / (self.action_high - self.action_low) * 2 - 1
        action_scaled = np.clip(action_scaled, -1, 1)
        
        # Add to replay buffer
        self.replay_buffer.add(state, action_scaled, reward, next_state, done)
        
        # Learn from experiences if buffer has enough samples
        if len(self.replay_buffer) >= self.min_batch_size:
            self.learn()
    
    def learn(self):
        """Update actor and critic networks using experience replay."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # ============ Update Critic ============
        # Calculate target Q-values
        with torch.no_grad():
            next_actions_residual = self.actor_target(next_states)
            if self._hover_tensor is None:
                self._hover_tensor = torch.from_numpy(self.hover_action).float().to(self.device)
            z_index, vz_index = self._state_layout(next_states.shape[1])
            next_z = next_states[:, z_index]
            if vz_index is not None:
                next_vz = next_states[:, vz_index]
            else:
                next_vz = torch.zeros_like(next_z)
            p_correction = (self.vz_gain * (-next_vz) + self.z_gain * (self.target_z - next_z)).unsqueeze(1)
            next_full_actions = next_actions_residual * self.delta_scale + self._hover_tensor + p_correction
            next_actions_scaled = (next_full_actions / self.action_high) * 2.0 - 1.0
            next_actions_scaled = torch.clamp(next_actions_scaled, -1.0, 1.0)
            target_q_values = self.critic_target(next_states, next_actions_scaled)
            # Bellman target
            y = rewards + self.gamma * target_q_values * (1 - dones)
        
        # Current Q-values
        q_values = self.critic(states, actions)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(q_values, y)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ============ Update Actor ============
        # Actor outputs residual [-1,1]; convert to full action scale for critic
        if self._hover_tensor is None:
            self._hover_tensor = torch.from_numpy(self.hover_action).float().to(self.device)
        z_index, vz_index = self._state_layout(states.shape[1])
        z = states[:, z_index]
        if vz_index is not None:
            vz = states[:, vz_index]
        else:
            vz = torch.zeros_like(z)
        p_correction = (self.vz_gain * (-vz) + self.z_gain * (self.target_z - z)).unsqueeze(1)
        predicted_residual = self.actor(states)
        full_actions = predicted_residual * self.delta_scale + self._hover_tensor + p_correction
        actions_scaled = (full_actions / self.action_high) * 2.0 - 1.0  # same scale as replay buffer
        actions_scaled = torch.clamp(actions_scaled, -1.0, 1.0)
        actor_loss = -self.critic(states, actions_scaled).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ============ Soft Update Target Networks ============
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def reset_epsilon(self):
        """Reset epsilon to initial value at the start of training."""
        self.epsilon = 1.0



