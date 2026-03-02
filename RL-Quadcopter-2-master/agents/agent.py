import numpy as np
from collections import deque
import random
import copy



#Replay Buffer

class ReplayBuffer:
    """Fixed‑size experience replay buffer."""

    def __init__(self, buffer_size=100_000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, k=batch_size)
        states      = np.array([e[0] for e in batch])
        actions     = np.array([e[1] for e in batch])
        rewards     = np.array([e[2] for e in batch]).reshape(-1, 1)
        next_states = np.array([e[3] for e in batch])
        dones       = np.array([e[4] for e in batch], dtype=np.float32).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Ornstein‑Uhlenbeck Noise

class OUNoise:
    """Ornstein‑Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        dx = (self.theta * (self.mu - self.state)
              + self.sigma * np.random.randn(self.size))
        self.state += dx
        return self.state


# Neural‑network helpers (NumPy)


def _relu(z):
    return np.maximum(0, z)

def _relu_deriv(z):
    return (z > 0).astype(z.dtype)

def _tanh(z):
    return np.tanh(z)

def _tanh_deriv(z):
    """Derivative of tanh given *pre‑activation* z."""
    t = np.tanh(z)
    return 1 - t * t


class _Layer:
    """Single dense layer with He initialisation and Adam optimiser."""

    def __init__(self, fan_in, fan_out, lr=1e-3):
        # He init
        self.W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((1, fan_out))
        self.lr = lr
        # Adam state
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.t = 0  # time‑step counter for bias correction

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        return self.z

    def backward(self, dz, clip=1.0):
        """Return dx and update weights via Adam."""
        m = self.x.shape[0]
        dW = self.x.T @ dz / m
        db = dz.mean(axis=0, keepdims=True)
        dx = dz @ self.W.T

        # Clip gradients
        dW = np.clip(dW, -clip, clip)
        db = np.clip(db, -clip, clip)

        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # W
        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * dW ** 2
        mW_hat = self.mW / (1 - beta1 ** self.t)
        vW_hat = self.vW / (1 - beta2 ** self.t)
        self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)

        # b
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * db ** 2
        mb_hat = self.mb / (1 - beta1 ** self.t)
        vb_hat = self.vb / (1 - beta2 ** self.t)
        self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

        return dx

    def copy_from(self, other):
        self.W = other.W.copy()
        self.b = other.b.copy()

    def soft_update(self, other, tau):
        self.W = tau * other.W + (1 - tau) * self.W
        self.b = tau * other.b + (1 - tau) * self.b


class Actor:
    """Deterministic policy: maps state to action in [low, high]."""

    def __init__(self, state_size, action_size, action_low, action_high,
                 lr=1e-4, h1=256, h2=128):
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = action_high - action_low

        self.fc1 = _Layer(state_size, h1, lr)
        self.fc2 = _Layer(h1, h2, lr)
        self.fc3 = _Layer(h2, action_size, lr)

    def forward(self, state):
        self.z1 = self.fc1.forward(state)
        self.a1 = _relu(self.z1)
        self.z2 = self.fc2.forward(self.a1)
        self.a2 = _relu(self.z2)
        self.z3 = self.fc3.forward(self.a2)
        raw = _tanh(self.z3)  # ∈ [-1, 1]
        # Scale to [action_low, action_high]
        action = raw * (self.action_range / 2) + (self.action_low + self.action_range / 2)
        self.raw = raw
        return action

    def train(self, daction):
        """Back‑propagate the critic's action‑gradient through the actor."""
        # daction: ∂Q/∂a, shape (batch, action_size)
        # Chain through tanh: da_raw = daction * action_range/2
        d_raw = daction * (self.action_range / 2)
        # Through tanh
        dz3 = d_raw * _tanh_deriv(self.z3)
        da2 = self.fc3.backward(dz3)
        dz2 = da2 * _relu_deriv(self.z2)
        da1 = self.fc2.backward(dz2)
        dz1 = da1 * _relu_deriv(self.z1)
        self.fc1.backward(dz1)

    def soft_update(self, other, tau):
        self.fc1.soft_update(other.fc1, tau)
        self.fc2.soft_update(other.fc2, tau)
        self.fc3.soft_update(other.fc3, tau)

    def copy_from(self, other):
        self.fc1.copy_from(other.fc1)
        self.fc2.copy_from(other.fc2)
        self.fc3.copy_from(other.fc3)

class Critic:
    """Q‑function approximator: (s, a) → scalar Q‑value."""

    def __init__(self, state_size, action_size, lr=1e-3, h1=256, h2=128):
        # Actions enter after the first hidden layer (as in original DDPG)
        self.fc1 = _Layer(state_size, h1, lr)
        self.fc2 = _Layer(h1 + action_size, h2, lr)
        self.fc3 = _Layer(h2, 1, lr)

    def forward(self, state, action):
        self.z1 = self.fc1.forward(state)
        self.a1 = _relu(self.z1)
        self.sa = np.hstack([self.a1, action])     # concat state‑features + action
        self.z2 = self.fc2.forward(self.sa)
        self.a2 = _relu(self.z2)
        self.z3 = self.fc3.forward(self.a2)         # linear output
        return self.z3

    def train(self, td_error):
        """Back‑propagate TD error. Returns dQ/da (needed by actor)."""
        # td_error shape: (batch, 1)
        dz3 = td_error  # ∂L/∂Q = td_error (from MSE derivative)
        da2 = self.fc3.backward(dz3)
        dz2 = da2 * _relu_deriv(self.z2)
        dsa = self.fc2.backward(dz2)
        # Split gradient for state‑features and action
        h1_size = self.a1.shape[1]
        da1 = dsa[:, :h1_size]
        daction = dsa[:, h1_size:]  # ∂Q/∂a  — used to train actor
        dz1 = da1 * _relu_deriv(self.z1)
        self.fc1.backward(dz1)
        return daction

    def soft_update(self, other, tau):
        self.fc1.soft_update(other.fc1, tau)
        self.fc2.soft_update(other.fc2, tau)
        self.fc3.soft_update(other.fc3, tau)

    def copy_from(self, other):
        self.fc1.copy_from(other.fc1)
        self.fc2.copy_from(other.fc2)
        self.fc3.copy_from(other.fc3)

class Agent:

    def __init__(self, task, gamma=0.99, tau=0.001,
                 actor_lr=1e-4, critic_lr=1e-3,
                 buffer_size=100_000, batch_size=64,
                 explore_mu=0., explore_theta=0.15, explore_sigma=0.2):
        self.task = task
        self.state_size  = task.state_size
        self.action_size = task.action_size
        self.action_low  = task.action_low
        self.action_high = task.action_high

        self.gamma = gamma
        self.tau   = tau
        self.batch_size = batch_size

        # --- Actor (main + target) ---
        self.actor = Actor(self.state_size, self.action_size,
                           self.action_low, self.action_high, lr=actor_lr)
        self.actor_target = Actor(self.state_size, self.action_size,
                                  self.action_low, self.action_high, lr=actor_lr)
        self.actor_target.copy_from(self.actor)

        # --- Critic (main + target) ---
        self.critic = Critic(self.state_size, self.action_size, lr=critic_lr)
        self.critic_target = Critic(self.state_size, self.action_size, lr=critic_lr)
        self.critic_target.copy_from(self.critic)

        #Replay buffer 
        self.memory = ReplayBuffer(buffer_size)

        #  Exploration noise
        self.noise = OUNoise(self.action_size,
                             mu=explore_mu,
                             theta=explore_theta,
                             sigma=explore_sigma)

        #Episode bookkeeping 
        self.best_score = -np.inf
        self.score = 0.
        self.noise_scale = explore_sigma   # for logging compatibility
        self.reset_episode()

    
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        self.noise.reset()
        state = self.task.reset()
        return state


    def act(self, state):
        """Select action = μ(s) + noise, clipped to valid range."""
        action = self.actor.forward(state.reshape(1, -1))[0]
        action += self.noise.sample()
        # Clip and enforce minimum rotor speed to avoid divide‑by‑zero in sim
        return np.clip(action, max(self.action_low, 1.0), self.action_high)

    
    def step(self, state, action, reward, next_state, done):
        """Store transition and learn from a mini‑batch."""
        self.memory.add(state, action, reward, next_state, done)
        self.total_reward += reward
        self.count += 1

        if len(self.memory) >= self.batch_size:
            self._learn_batch()

        if done:
            self.score = self.total_reward / float(self.count) if self.count else 0.
            if self.score > self.best_score:
                self.best_score = self.score


    def _learn_batch(self):
        """Sample a mini‑batch and do one gradient step for actor & critic."""
        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        # ----- Critic update -----
        # y = r + γ Q_target(s', μ_target(s'))
        next_actions = self.actor_target.forward(next_states)
        Q_targets_next = self.critic_target.forward(next_states, next_actions)
        y = rewards + self.gamma * Q_targets_next * (1 - dones)

        Q_predicted = self.critic.forward(states, actions)
        td_error = (Q_predicted - y)                      # shape (batch, 1)
        td_error_clipped = np.clip(td_error, -1, 1)       # Huber‑like clamp
        daction = self.critic.train(td_error_clipped)      # returns ∂Q/∂a

        # ----- Actor update -----
        # We want to *maximise* Q, so ascend: pass –∂Q/∂a (negative for ascent)
        self.actor.forward(states)        # regenerate activations for this batch
        self.actor.train(-daction)

        # ----- Soft‑update targets -----
        self.actor_target.soft_update(self.actor, self.tau)
        self.critic_target.soft_update(self.critic, self.tau)
