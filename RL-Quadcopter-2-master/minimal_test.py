"""Minimal test to diagnose why NaN persists in notebook but not in debug_test.py"""
import sys
import numpy as np
import warnings
from agents.agent import PolicySearch_Agent
from task import Task

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Exact same setup as notebook training cell
num_episodes = 100
target_pos = np.array([0., 0., 10.])
init_pose = np.array([0., 0., 0., 0., 0., 0.])
init_velocities = np.array([0., 0., 0.])
init_angle_velocities = np.array([0., 0., 0.])
runtime = 5.

task = Task(init_pose=init_pose, init_velocities=init_velocities,
            init_angle_velocities=init_angle_velocities, runtime=runtime,
            target_pos=target_pos)
agent = PolicySearch_Agent(task)

episode_rewards = []
valid_episodes = 0
nan_count = 0

print("Testing exact notebook code...")
for i_episode in range(1, num_episodes + 1):
    state = agent.reset_episode()
    
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(reward, done)
        state = next_state
        
        if done:
            # Check if score is valid
            if np.isfinite(agent.score):
                episode_rewards.append(agent.score)
                valid_episodes += 1
            else:
                episode_rewards.append(np.nan)
                nan_count += 1
                if nan_count <= 3:
                    print(f"  Episode {i_episode}: NaN detected, total_reward={agent.total_reward}, count={agent.count}")
            
            if i_episode % 20 == 0:
                valid_so_far = sum(1 for r in episode_rewards if np.isfinite(r))
                print(f"Episode {i_episode:3d}: Score={agent.score:10.4f}, Best={agent.best_score:7.4f}, Valid={valid_so_far}/{i_episode}, Noise={agent.noise_scale:.4f}")
            break

print(f"\nFinal stats:")
print(f"Valid episodes: {valid_episodes}/{num_episodes}")
print(f"NaN episodes: {nan_count}/{num_episodes}")
print(f"Agent state: nan_episode_count={agent.nan_episode_count}")
