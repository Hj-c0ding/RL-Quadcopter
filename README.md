# RL-Quadcopter
A reinforcement learning agent trained to fly a simulated quadcopter, built using DDPG (Deep Deterministic Policy Gradient) in PyTorch.
Overview
This project trains a quadcopter agent in two stages:

1. Hover — learn to maintain a fixed altitude
2. Point Navigation — fly to a target position in 3D space

The physics simulation models real quadcopter dynamics including rotor thrust, drag, angular momentum, and propeller wind effects.
