import sys
import numpy as np
import warnings
from agents.agent import PolicySearch_Agent
from task import Task

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Task configuration
num_episodes = 60  # Run 60 episodes to see where NaN appears
target_pos = np.array([0., 0., 10.])
init_pose = np.array([0., 0., 0., 0., 0., 0.])
init_velocities = np.array([0., 0., 0.])
init_angle_velocities = np.array([0., 0., 0.])
runtime = 5.

task = Task(init_pose=init_pose, init_velocities=init_velocities,
            init_angle_velocities=init_angle_velocities, runtime=runtime,
            target_pos=target_pos)
agent = PolicySearch_Agent(task)

print("Initial weights shape:", agent.w.shape)
print("Max initial weight:", np.max(np.abs(agent.w)))
print()

for i_episode in range(1, num_episodes + 1):
    print(f"\n===== EPISODE {i_episode} =====")
    state = agent.reset_episode()
    step_count = 0
    
    while True:
        # Check state before action
        if not np.all(np.isfinite(state)):
            print(f"  STEP {step_count}: State contains NaN before act()")
            print(f"    State[:6] = {state[:6]}")
        
        action =agent.act(state)
        
        # Check action after act
        if not np.all(np.isfinite(action)):
            print(f"  STEP {step_count}: Action contains NaN from act()")
            print(f"    Action = {action}")
            print(f"    Max weights = {np.max(np.abs(agent.w))}")
        
        next_state, reward, done = task.step(action)
        
        # Check outputs from step
        if not np.isfinite(reward):
            print(f"  STEP {step_count}: Reward is NaN")
            print(f"    Total reward so far: {agent.total_reward}")
            print(f"    Sim pose: {task.sim.pose}")
            print(f"    Sim velocity: {task.sim.v}")
        
        if not np.all(np.isfinite(next_state)):
            print(f"  STEP {step_count}: Next state contains NaN")
            print(f"    Next state[:6] = {next_state[:6]}")
        
        agent.step(reward, done)
        state = next_state
        step_count += 1
        
        if done:
            print(f"Episode {i_episode}: score={agent.score:.4f}, best={agent.best_score:.4f}, " +
                  f"noise={agent.noise_scale:.4f}, max_weight={np.max(np.abs(agent.w)):.4f}, " +
                  f"nan_count={agent.nan_episode_count}")
            break

print("\nDebug test complete!")
