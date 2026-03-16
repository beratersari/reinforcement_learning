# Pac-Man RL Simulation (Multi-Agent)

A customizable Pac-Man simulation environment (30x30 grid) for training and evaluating multi-agent Reinforcement Learning (RL) models. This project supports **Q-Learning**, **MADDPG**, **Shared DQN**, and **QMIX** (Factorized Q-Learning) agents controlling the ghosts to capture Pac-Man.

## Features

- **Multi-Agent RL**: Supports 2 or 4 ghosts learning cooperatively or independently.
- **Models**:
  - **Q-Learning**: Independent tabular Q-learning per ghost.
  - **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient (Actor-Critic).
  - **Shared DQN**: Deep Q-Network with shared weights and replay buffer, using ghost IDs for role differentiation.
  - **QMIX**: Factorized Multi-Agent Q-Learning with centralized training and decentralized execution. Uses a mixing network to combine individual Q-values into joint Q-value.
- **Coordination**: Built-in mechanisms for encirclement and trapping strategies.
- **Reward Shaping**: Team-based rewards to encourage spacing, mobility reduction, and discourage clustering/line formations.
- **Anti-Pattern Logic**: Hard overrides to prevent infinite loops (oscillations) and ghost-on-ghost collisions.
- **Maps**: 6 procedural and classic map layouts.
- **Logging**: Detailed move logging for behavior analysis.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a Model
Train **Shared DQN** (recommended) on Map 0 for 1000 episodes:
```bash
python train_ghosts.py --episodes 1000 --model dqn --save-dir models_dqn/
```

Train **Shared DQN** on random maps:
```bash
python train_ghosts.py --episodes 1000 --model dqn --train-random-map --save-dir models_random/
```

Train **Q-Learning** (single map):
```bash
python train_ghosts.py --episodes 1000 --model qlearning --save-dir models_ql/
```

Train **Q-Learning** on random maps:
```bash
python train_ghosts.py --episodes 1000 --model qlearning --train-random-map --save-dir models_ql_random/
```

Train **MADDPG** (single map):
```bash
python train_ghosts.py --episodes 1000 --model maddpg --save-dir models_maddpg/
```

Train **MADDPG** on random maps:
```bash
python train_ghosts.py --episodes 1000 --model maddpg --train-random-map --save-dir models_maddpg_random/
```

Train **QMIX** (single map):
```bash
python train_ghosts.py --episodes 1000 --model qmix --save-dir models_qmix/
```

Train **QMIX** on random maps:
```bash
python train_ghosts.py --episodes 1000 --model qmix --train-random-map --save-dir models_qmix_random/
```

### 3. Evaluate and Watch
Watch the trained agents play (no training):
```bash
python train_ghosts.py --episodes 10 --model dqn --load-dir models_dqn/ --eval-only --render --fps 15
```

### 4. Log Moves for Analysis
Run evaluation and save all moves to `evaluation_moves.txt`:
```bash
python train_ghosts.py --episodes 50 --model dqn --load-dir models_dqn/ --eval-only --log-moves
```

---

## Configuration

Hyperparameters are defined in `config.json`. You can override them or provide a custom config file with `--config`.

### Default Config Structure
```json
{
  "dqn": {
    "lr": 0.0005,
    "gamma": 0.95,
    "batch_size": 128,
    "buffer_size": 100000,
    "update_every": 4,
    "tau": 0.001,
    "hidden_sizes": [128, 128]
  },
  "qmix": {
    "lr": 0.0005,
    "gamma": 0.95,
    "batch_size": 128,
    "buffer_size": 100000,
    "update_every": 4,
    "tau": 0.005,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "hidden_sizes": [128, 128],
    "mixing_hidden": 64
  },
  "rewards": {
    "collision": 100,
    "pellets_done": -100,
    "closer": 1,
    "farther": -1,
    "invalid": -0.5
  },
  "team_rewards": {
    "mobility": 1.5,
    "spread": 0.5,
    "cluster_penalty": -0.6,
    "line_penalty": -1.2,
    "overlap_penalty": -0.4
  }
}
```

---

## Models Explained

### 1. Q-Learning (Independent)
- **Type**: Tabular.
- **Structure**: Each ghost has its own Q-table.
- **Pros**: Simple, interpretable.
- **Cons**: Does not scale well to large state spaces; struggles with coordination without explicit shaping.

### 2. MADDPG (Multi-Agent Actor-Critic)
- **Type**: Deep RL (Continuous action space mapped to discrete).
- **Structure**: Each ghost has an Actor and Critic network.
- **Mechanism**: Centralized training, decentralized execution.
- **Pros**: Handles multi-agent dynamics well.

### 3. Shared DQN (Deep Q-Network)
- **Type**: Deep RL.
- **Structure**: Single Q-network shared by all ghosts.
- **Input**: State vector + One-hot encoded **Ghost ID**.
- **Mechanism**: Experience replay buffer is shared. Ghosts learn from each other's experiences while developing distinct roles via the ID input.
- **Pros**: Data efficient, faster convergence, reduced memory usage.

### 4. QMIX (Factorized Multi-Agent Q-Learning)
- **Type**: Deep Multi-Agent RL.
- **Structure**: Per-ghost Q-networks (individual utility functions) + shared **Mixing Network**.
- **Mechanism**: 
  - **Decentralized Execution**: Each ghost selects actions using its own Q-network independently.
  - **Centralized Training**: A mixing network combines individual Q-values into a joint Q-value using a hypernetwork that generates weights from global state.
  - **Monotonicity**: Mixing network uses absolute weights to ensure `argmax(Q_joint) = argmax(individual Qs)`, enabling decentralized execution.
- **Input**: Individual state per ghost; global state (concatenation) for mixing network.
- **Pros**: 
  - Strong multi-agent coordination via joint Q-value optimization.
  - Scales to larger teams better than independent Q-learning.
  - Maintains decentralized execution efficiency.
- **Cons**: More complex than Shared DQN; requires global state during training.

---

## Environment Mechanics

### State Space
Each ghost observes a local state (9-dim tuple):
- Distance to Pac-Man (discretized)
- Direction to Pac-Man
- Distance & Direction to nearest ghost
- Valid moves (walls)
- Nearby pellets

**For DQN**: The state is augmented with a one-hot vector representing the ghost ID (e.g., `[0, 1, 0, 0]` for Ghost 1).

### Partial Observability
Controlled via `observation` config section:
- **`range`**: Manhattan distance threshold for seeing Pac-Man (default: 30 = full observability)
- **`shared_pacman`**: 
  - `false` (default): Each ghost only observes PM if within their individual range
  - `true`: Team mode - if ANY ghost within range sees PM, ALL ghosts get full PM info

**Examples:**
```json
// Individual partial observability (each ghost limited by range)
"observation": { "range": 8, "shared_pacman": false }

// Team shared observability (any ghost within range shares PM with all)
"observation": { "range": 8, "shared_pacman": true }
```

### Anti-Pattern Logic (Hardcoded)
To ensure robust behavior, the environment enforces:
1. **No Overlaps**: Ghosts cannot move onto a cell occupied by another ghost (physically blocking).
2. **No Oscillations**: If a ghost repeats a position pattern (e.g., A-B-A-B) within a short window, it is forced to pick a different move.
3. **No Line Formations**: If ghosts form a straight line (horizontal/vertical), they are forced to break formation to prevent "train" chasing.

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `--model` | Model type: `qlearning`, `maddpg`, `dqn`, `qmix`. |
| `--episodes` | Number of episodes to run. |
| `--ghosts` | Number of ghosts (2 or 4). |
| `--train-map` | Map index (0-5) for training. |
| `--train-random-map` | Train on a random map each episode (ignores `--train-map`). |
| `--save-dir` | Directory to save checkpoints. |
| `--load-dir` | Directory to load models from. |
| `--eval-only` | Disable training (inference mode). |
| `--render` | Enable visual window. |
| `--log-moves` | Save move logs to `evaluation_moves.txt`. |

---

## Project Structure
- `pacman_game.py`: Core game logic, map generation, and rendering.
- `ghost_rl.py`: Agent implementations (DQN, MADDPG, Q-Learning).
- `train_ghosts.py`: Training loop, environment wrapper, and CLI entry point.
- `config.json`: Hyperparameters.
