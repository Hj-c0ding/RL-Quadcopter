import numpy as np
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
        states = np.vstack([e[0] for e in experiences])
        actions = np.vstack([e[1] for e in experiences])
        rewards = np.vstack([e[2] for e in experiences])
        next_states = np.vstack([e[3] for e in experiences])
        dones = np.vstack([e[4] for e in experiences]).astype(np.uint8)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class NeuralNetwork:
    """Simple 3-layer neural network with ReLU activations (NumPy implementation)"""
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0005):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        
        # Cache for backpropagation
        self.cache = {}
    
    def forward(self, X):
        """Forward pass through network"""
        # Layer 1
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)  # ReLU activation
        
        # Layer 2
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = np.maximum(0, Z2)  # ReLU activation
        
        # Layer 3
        Z3 = np.dot(A2, self.W3) + self.b3
        
        # Cache for backprop
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3}
        
        return Z3
    
    def backward(self, dZ3, batch_size):
        """Backward pass through network with gradient descent update"""
        # Extract cached values
        X = self.cache['X']
        Z1 = self.cache['Z1']
        A1 = self.cache['A1']
        Z2 = self.cache['Z2']
        A2 = self.cache['A2']
        
        # Output layer gradients
        dW3 = np.dot(A2.T, dZ3) / batch_size
        db3 = np.sum(dZ3, axis=0, keepdims=True) / batch_size
        
        # Hidden layer 2 gradients
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (Z2 > 0)  # ReLU derivative
        dW2 = np.dot(A1.T, dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        
        # Hidden layer 1 gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
        dW1 = np.dot(X.T, dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
        
        # Gradient clipping for stability
        dW1 = np.clip(dW1, -1, 1)
        dW2 = np.clip(dW2, -1, 1)
        dW3 = np.clip(dW3, -1, 1)
        db1 = np.clip(db1, -1, 1)
        db2 = np.clip(db2, -1, 1)
        db3 = np.clip(db3, -1, 1)
        
        # Update weights with gradient descent
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
    
    def copy_weights(self, other):
        """Copy weights from another network"""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


class Agent:
    """Q-Learning Agent with continuous action space"""
    def __init__(self, task, hidden_size=128, learning_rate=0.0005, gamma=0.99):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        
        # Q-Network (estimates Q-values for continuous actions)
        self.network = NeuralNetwork(
            input_size=self.state_size,
            hidden_size=hidden_size,
            output_size=self.action_size,
            learning_rate=learning_rate
        )
        
        # Target Q-Network (for stability)
        self.target_network = NeuralNetwork(
            input_size=self.state_size,
            hidden_size=hidden_size,
            output_size=self.action_size,
            learning_rate=learning_rate
        )
        self._copy_weights()
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size=10000)
        self.batch_size = 64
        self.train_counter = 0
        self.train_interval = 4
        self.update_target_interval = 1000
        
        # Exploration parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.best_score = -np.inf
        self.noise_scale = 0.1  # For compatibility with notebook
        
        self.reset_episode()
    
    def _copy_weights(self):
        """Copy main network weights to target network"""
        self.target_network.copy_weights(self.network)
    
    def act(self, state):
        """Select action using continuous Q-learning policy"""
        # Reshape state for network
        state_input = state.reshape(1, -1)
        
        # Get Q-values from network (interpreted as action values)
        q_values = self.network.forward(state_input)[0]
        
        # Scale Q-values to action range (0 to 900)
        # Map from [-1, 1] range to [action_low, action_high]
        action = np.tanh(q_values) * (self.action_range / 2) + (self.action_low + self.action_range / 2)
        
        # Epsilon-greedy exploration: add noise with probability epsilon
        if random.random() < self.epsilon:
            # Add Gaussian noise for exploration
            action += np.random.normal(0, self.noise_scale * self.action_range, size=self.action_size)
        
        # Clip to valid rotor speeds
        action = np.clip(action, self.action_low, self.action_high)
        # Ensure minimum rotor speed
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
        
        # Store experience (normalize reward for stability)
        normalized_reward = reward / 100.0
        self.memory.add(state, action, normalized_reward, next_state, float(done))
        
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
        
        # Forward pass through main network
        q_values = self.network.forward(states)
        
        # Forward pass through target network for next states
        next_q_values = self.target_network.forward(next_states)
        
        # For continuous action space: compute max Q-value for next states
        # We use the output as our Q-values
        max_next_q = np.max(next_q_values, axis=1, keepdims=True)
        
        # Compute target Q-values using temporal difference
        target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute TD error (gradient of MSE loss)
        batch_size = states.shape[0]
        dZ3 = (q_values - target_q) * 2 / batch_size  # MSE derivative
        
        # Backward pass (update network weights)
        self.network.backward(dZ3, batch_size)
    
    def learn(self):
        """Called at end of episode to update learning parameters"""
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        
        # Track best score
        if self.score > self.best_score:
            self.best_score = self.score
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.noise_scale = self.epsilon  # Update noise scale for logging
