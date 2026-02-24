import numpy as np


class PolicySearch_Agent():
    """Improved Policy Search Agent for Quadcopter Takeoff and Hovering Task.
    
    This agent uses a linear policy with adaptive noise, weight clipping, and
    NaN-safe mechanisms to learn how to control the quadcopter for takeoff and hovering.
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

        # Initialize weights with very small random values to avoid divergence
        # Small initial weights ensure mild early actions while agent explores
        # This is critical for physics stability - large actions cause simulation NaN
        self.w = np.random.normal(size=(self.state_size, self.action_size), scale=0.01)

        # Score tracking and best policy parameters
        self.best_w = self.w.copy()
        self.best_score = -np.inf
        
        # Adaptive noise and learning parameters
        self.noise_scale = 0.1  # Start with small noise for careful exploration
        self.learning_rate = 0.01  # Learning rate for weight updates
        self.noise_decay = 0.992  # Decay noise more aggressively over time
        self.max_weight_magnitude = 5.0  # Tighter bound prevent extreme actions
        
        # Episode tracking and convergence monitoring
        self.total_reward = 0.0
        self.count = 0
        self.score = 0.0
        self.consecutive_no_improve = 0  # Track episodes without improvement
        self.nan_episode_count = 0  # Track invalid episodes
        self.last_valid_w = self.w.copy()  # Backup of last valid weights

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
        
        Uses a linear policy centered around hovering thrust (450 RPM):
        action = 450 + state @ weights
        
        Clips output to valid action range [action_low, action_high]
        Handles NaN/Inf gracefully by using fallback hover thrust.
        
        Params
        ======
            state: current state (observation)
            
        Returns
        =======
            action: clipped action for each rotor (4-dimensional)
        """
        # Linear policy centered on hovering thrust (450 RPM)
        # This ensures exploration happens around a reasonable baseline
        hover_thrust = 450.0
        action = hover_thrust + np.dot(state, self.w)
        
        # Safety: Handle any NaN/Inf by reverting to backup weights
        if not np.all(np.isfinite(action)):
            self.w = self.last_valid_w.copy()
            action = hover_thrust + np.dot(state, self.w)
        
        # Handle remaining NaN/Inf with fallback to pure hovering
        action = np.nan_to_num(action, nan=hover_thrust, posinf=self.action_high, neginf=self.action_low)
        
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
        
        Uses random policy search with adaptive noise and safety mechanisms:
        - If performance improved: reduce noise (fine-tune)
        - If performance worsened: increase noise (explore more)
        - Resets noise if stuck at max for 20+ episodes
        - Clips weights to prevent explosion
        - Skips updates if score is invalid (NaN/Inf)
        """
        # Compute average reward for this episode
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        
        # Check if score is valid
        if not self._is_valid_score(self.score):
            self.nan_episode_count += 1
            self.consecutive_no_improve += 1
            # Skip weight update for invalid episodes
            return
        
        # Save as last valid weights
        self.last_valid_w = self.w.copy()
        
        # Check if this is the best score so far
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w.copy()
            # Found better solution: reduce noise for fine-tuning
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
            self.consecutive_no_improve = 0
        else:
            # Not better: revert to best weights and increase noise moderately for exploration
            self.w = self.best_w.copy()
            self.noise_scale = min(1.5 * self.noise_scale, 2.0)  # More conservative: max 2.0 instead of 3.2
            self.consecutive_no_improve += 1
        
        # Smart reset: if stuck at max noise for too long, reset to explore differently
        if self.consecutive_no_improve >= 15 and self.noise_scale >= 1.9:  # Lower threshold
            self.noise_scale = 0.05  # Reset to very small noise
            self.consecutive_no_improve = 0
        
        # Apply time-based noise decay (gradually reduce exploration)
        self.noise_scale *= self.noise_decay
        
        # Update weights with noise for next episode
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)
        
        # Clip weights to prevent explosion
        self._clip_weights()
    
    def _is_valid_score(self, score):
        """Check if score is valid (not NaN or Inf).
        
        Params
        ======
            score: reward score to validate
            
        Returns
        =======
            bool: True if score is finite, False otherwise
        """
        return np.isfinite(score)
    
    def _clip_weights(self):
        """Clip weight magnitudes to prevent unbounded growth.
        
        Ensures no single weight exceeds max_weight_magnitude in absolute value.
        This prevents actions from becoming infinity.
        """
        # Clip each weight to [-max_weight_magnitude, +max_weight_magnitude]
        self.w = np.clip(self.w, -self.max_weight_magnitude, self.max_weight_magnitude)
        
        # Also clip best weights to keep them consistent
        self.best_w = np.clip(self.best_w, -self.max_weight_magnitude, self.max_weight_magnitude)