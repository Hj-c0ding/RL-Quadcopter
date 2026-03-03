# Quadcopter Agent:
---

## 1) Big Picture

**continuous-control Actor–Critic** setup:

- **Actor**: takes a state and outputs 4 rotor speeds (action).
- **Critic**: estimates how good a `(state, action)` pair is (Q-value).
- **Replay Buffer**: stores experiences and samples random mini-batches for stable learning.
- **Target Networks**: slow-moving copies of actor and critic used to stabilize training targets.
- **OU Noise**: adds smooth exploration noise to actions.

---

## 2) Reward Function in `task.py`

In `task.py`, reward is:

$$
\text{reward} = 1 - 0.3\cdot\sum\left|\text{pose}_{xyz} - \text{target}_{xyz}\right|
$$

Interpretation:

- If the drone is close to target position, reward is high.
- If it is far, reward drops (can become negative).
- In `step()`, reward is accumulated across `action_repeat = 3` simulator steps.

So the policy learns: **pick rotor speeds that reduce position error over time**.

---

## 3) Data Flow + Exploration

### Replay Buffer (`ReplayBuffer`)

Stores transitions:

- `(state, action, reward, next_state, done)`

Why random sampling helps:

- Breaks temporal correlation.
- Improves optimization stability.

### OU Noise (`OUNoise`)

Used in `Agent.act(state)`:

1. Actor predicts action.
2. Noise is added for exploration.
3. Action is clipped to valid rotor speed range.

This gives exploration that is smoother than pure white noise.

---

## 4) Actor Network

Class: `Actor`

Architecture:

- `state -> Dense(256) -> ReLU -> Dense(128) -> ReLU -> Dense(4) -> tanh`
- `tanh` output in `[-1, 1]` is scaled to `[action_low, action_high]`.

Training idea:

- Critic provides gradient $\partial Q/\partial a$.
- Actor updates weights to **increase** Q-values.
- In code this appears as passing `-daction` to gradient descent-based updates.

---

## 5) Critic Network

Class: `Critic`

Architecture:

- State goes through first hidden layer.
- Action is concatenated with state features in second layer.
- Final output is one scalar Q-value.

Target used for learning:

$$
y = r + \gamma \cdot Q_{\text{target}}\left(s',\mu_{\text{target}}(s')\right)\cdot(1-done)
$$

Then TD error is formed and used for backprop:

$$
\delta = Q_{\text{pred}} - y
$$

The code clips TD error before update for stability.

---

## 6) Learning Loop in `Agent`

### `step(state, action, reward, next_state, done)`

- Save transition to replay buffer.
- Update episode reward counters.
- If enough samples exist, call `_learn_batch()`.

### `_learn_batch()`

1. Sample mini-batch from replay buffer.
2. Compute critic targets using target actor + target critic.
3. Update critic from TD error.
4. Update actor using critic action gradients.
5. Soft-update target networks:

$$
\theta_{\text{target}} \leftarrow \tau\theta_{\text{main}} + (1-\tau)\theta_{\text{target}}
$$

---

## 7) Code Map of `agent.py`

- **`ReplayBuffer`**: memory for off-policy learning.
- **`OUNoise`**: temporally correlated exploration.
- **`_Layer` + activations**: NumPy dense layers + Adam optimizer logic.
- **`Actor`**: deterministic policy network.
- **`Critic`**: Q-value network.
- **`Agent`**: orchestration (`act`, `step`, `_learn_batch`, episode tracking).

---

## 8) Key Hyperparameters (quick reference)

- `gamma` (`0.99`): discount factor (future reward importance).
- `tau` (`0.001`): target-network soft-update speed.
- `actor_lr` / `critic_lr`: learning rates.
- `buffer_size`, `batch_size`: replay settings.
- `explore_theta`, `explore_sigma`: OU noise behavior.

---
