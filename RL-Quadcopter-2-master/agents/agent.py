import numpy as np


class PolicySearch_Agent():
    """Improved Policy Search Agent for Quadcopter Takeoff and Hovering Task.
    
    This agent uses a linear policy with adaptive noise and learning rate to learn
    how to control the quadcopter for takeoff and hovering.
    """
    
    def __init__(self, task):
        """Initialize the agent.
        
        Params
        ======
            task: Task object defining the environment
        """
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # Initialize weights using Xavier/Glorot-style scaling
        # This provides better initial weight distribution than simple normal
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),
            scale=(self.action_range / (2 * np.sqrt(self.state_size))))

        # Score tracking and best policy parameters
        self.best_w = self.w.copy()
        self.best_score = -np.inf
        
        # Adaptive noise and learning parameters
        self.noise_scale = 0.5  # Start with moderate noise
        self.learning_rate = 0.01  # Learning rate for weight updates
        self.noise_decay = 0.995  # Decay noise over time
        
        # Episode tracking
        self.total_reward = 0.0
        self.count = 0
        self.score = 0.0

    def reset_episode(self):
        """Reset episode tracking variables and get initial state.
        
        Returns
        =======
            state: initial state for the episode
        """
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def act(self, state):
        """Choose action based on current state and learned policy.
        
        Uses a linear policy: action = state @ weights
        Clips output to valid action range [action_low, action_high]
        
        Params
        ======
            state: current state (observation)
            
        Returns
        =======
            action: clipped action for each rotor (4-dimensional)
        """
        # Linear policy
        action = np.dot(state, self.w)
        
        # Clip to valid action range
        action = np.clip(action, self.action_low, self.action_high)
        
        return action

    def step(self, reward, done):
        """Update agent based on reward received.
        
        Params
        ======
            reward: reward signal from the environment
            done: boolean indicating if episode is finished
        """
        # Accumulate reward for the episode
        self.total_reward += reward
        self.count += 1

        # Learn when episode is done
        if done:
            self.learn()

    def learn(self):
        """Update policy weights based on episode performance.
        
        Uses random policy search with adaptive noise:
        - If performance improved: reduce noise (fine-tune)
        - If performance worsened: increase noise (explore more)
        
        Also applies noise decay to gradually reduce exploration over time.
        """
        # Compute average reward for this episode
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        
        # Check if this is the best score so far
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w.copy()
            # Found better solution: reduce noise for fine-tuning
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            # Not better: revert to best weights and increase noise for exploration
            self.w = self.best_w.copy()
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        
        # Apply time-based noise decay (gradually reduce exploration)
        self.noise_scale *= self.noise_decay
        
        # Update weights with noise for next episode
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)