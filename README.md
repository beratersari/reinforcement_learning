# Pac-Man RL Simulation (QLearning)

A Pac-Man simulation environment (30x30 grid) for training Q-learning ghost agents and evaluating them across multiple maps.

## Contents

- [Quick Start](#quick-start)
- [Maps](#maps)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation (Test Only)](#evaluation-test-only)
- [Render / Watch Mode](#render--watch-mode)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [State + Rewards](#state--rewards)
- [Models](#models)

---

## Quick Start

```bash
pip install -r requirements.txt

# Train Q-learning on map 0 (default)
python train_ghosts.py --episodes 1000 --save-dir models/

# Evaluate Q-learning on a different map
python train_ghosts.py --episodes 200 --test-map 2 --load-dir models/ --eval-only --render

# Train MADDPG
python train_ghosts.py --episodes 1000 --model maddpg --save-dir models_maddpg/
```

---

## Maps

Six auto-generated maps are available (index 0-5). Each run uses the same index ordering:

| Index | Type | Description |
|-------|------|-------------|
| 0 | Random | Sparse random walls, open corridors |
| 1 | Maze | Recursive backtracker maze (single-path structure) |
| 2 | Classic | Classic-style corridors + obstacles |
| 3 | Random | New random layout |
| 4 | Maze | New maze variant |
| 5 | Classic | New classic variant |

Select maps using:
- `--train-map N`
- `--test-map N`

---

## Configuration

All hyperparameters are stored in [config.json](config.json). CLI arguments are reserved for **game**, **training**, and **model** options.

### Model Selection (CLI)

Set algorithm via `--model` argument:

```bash
# Q-Learning (default)
python train_ghosts.py --episodes 1000 --model qlearning --save-dir models_ql/

# MADDPG
python train_ghosts.py --episodes 1000 --model maddpg --save-dir models_maddpg/
```

### Q-Learning Example Config

```json
{
  "q_learning": {
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "use_noop": false
  }
}
```

### MADDPG Example Config

```json
{
  "maddpg": {
    "actor_lr": 0.0005,
    "critic_lr": 0.001,
    "gamma": 0.95,
    "tau": 0.01,
    "batch_size": 128,
    "buffer_size": 100000,
    "update_every": 1,
    "epsilon_start": 0.2,
    "epsilon_min": 0.02,
    "epsilon_decay": 0.995,
    "hidden_sizes": [128, 128]
  },
  "coordination": {
    "enabled": true,
    "vision_range": 8,
    "surround": true
  }
}
```

To use a custom config:

```bash
python train_ghosts.py --episodes 1000 --config my_config.json --save-dir models_custom/
```

---

## Training

```bash
# Basic training (Q-learning, map 0)
python train_ghosts.py --episodes 1000 --save-dir models/

# Train on a specific map
python train_ghosts.py --episodes 1000 --train-map 2 --save-dir models_map2/

# Train with 2 ghosts
python train_ghosts.py --ghosts 2 --episodes 1000 --save-dir models_2g/

# Continue training from saved models
python train_ghosts.py --load-dir models/ --episodes 500

# Train with MADDPG
python train_ghosts.py --episodes 1000 --model maddpg --save-dir models_maddpg/
```

---

## Evaluation (Test Only)

Use **eval-only mode** to test a trained model on a chosen map without training.

```bash
# Evaluate Q-learning on map 2 (no training)
python train_ghosts.py --episodes 200 --test-map 2 --load-dir models/ --eval-only

# Evaluate MADDPG on map 2 (set model.type=maddpg in config)
python train_ghosts.py --episodes 200 --test-map 2 --load-dir models_maddpg/ --eval-only

# Evaluate and render
python train_ghosts.py --episodes 200 --test-map 2 --load-dir models/ --eval-only --render
```

---

## Render / Watch Mode

```bash
# Watch training (every episode)
python train_ghosts.py --render --episodes 100 --fps 10

# Watch training every 10th episode
python train_ghosts.py --render-every 10 --episodes 500 --fps 15
```

You can also run the interactive game directly:

```bash
python pacman_game.py
```

Controls:
- **Menu**: `1-9` select map | `UP/DOWN` navigate | `ENTER` start
- **Game**: `R` restart | `N` next map | `M` menu | `ESC` quit

---

## CLI Reference

| Category | Flag | Description |
|----------|------|-------------|
| **Game** | `--episodes N` | Number of episodes |
| **Game** | `--ghosts N` | Number of ghosts: 2 or 4 |
| **Game** | `--pellets N` | Max pellets (default 100) |
| **Game** | `--time-limit N` | Pac-Man timeout in seconds (default 20) |
| **Map** | `--train-map N` | Training map index (0-5) |
| **Map** | `--test-map N` | Test map index (0-5) |
| **Train** | `--save-dir PATH` | Save model files here (Q-tables / MADDPG weights) |
| **Train** | `--load-dir PATH` | Load model files from here |
| **Train** | `--save-every N` | Save checkpoint every N episodes |
| **Train** | `--config PATH` | Config file path (default config.json) |
| **Model** | `--model TYPE` | Algorithm: qlearning or maddpg (default qlearning) |
| **Train** | `--eval-only` | Run evaluation only (requires --load-dir) |
| **Render** | `--render` | Show game window (all episodes) |
| **Render** | `--render-every N` | Render every N episodes |
| **Render** | `--fps N` | Render FPS (default 10) |

---

## Project Structure

```
.
├── pacman_game.py      # Simulation + map generator + visualization
├── ghost_rl.py         # Q-learning + MADDPG agents
├── train_ghosts.py     # Training + evaluation CLI
├── config.json         # Hyperparameters (model via --model CLI arg)
├── requirements.txt    # Dependencies
└── models/             # Saved models (Q-tables / MADDPG weights)
```

---

## State + Rewards

### Observability

Ghosts are **partially observable**: each ghost only sees a compact, local summary (distance + direction to Pac-Man, nearest ghost, immediate wall availability, nearby pellets). They do **not** receive the full grid layout or exact positions for every pellet/ghost.

### State Representation

Each ghost observes a compact 9-element tuple:

| Index | Description | Bins |
|-------|-------------|------|
| 0 | Distance to Pac-Man | 6 bins (0-1, 2-3, 4-7, 8-15, 16-29, 30+) |
| 1 | Direction to Pac-Man | 4 (UP/DOWN/LEFT/RIGHT) |
| 2 | Distance to nearest ghost | 6 |
| 3 | Direction to nearest ghost | 5 (4 dirs + none) |
| 4-7 | Can move UP/DOWN/LEFT/RIGHT | 2 each (binary) |
| 8 | Pellet nearby (≤3 cells) | 2 (binary) |

MADDPG uses the same state encoding but outputs **continuous actions** mapped to discrete moves via `argmax`.

### Rewards

| Event | Reward |
|-------|--------|
| ANY ghost catches Pac-Man | +100 (all ghosts) |
| Pac-Man eats all pellets | -100 (all ghosts) |
| Moved closer to Pac-Man | +1 |
| Moved away from Pac-Man | -1 |
| Invalid move (wall) | -0.5 |
| Repeated pattern (A-B-A-B or stuck) | -0.25 |

**Team shaping rewards (anti-line / anti-cluster):**
- **Mobility reduction**: +1.5 per step when Pac-Man loses valid escape moves
- **Adjacent block**: +0.75 per ghost adjacent to Pac-Man (blocking)
- **Spread bonus**: +0.5 when average inter-ghost distance increases
- **Cluster penalty**: -0.6 for ghosts within 1 cell of each other
- **Line penalty**: -1.2 if all ghosts align on one row or column
- **Overlap penalty**: -0.4 if two ghosts overlap in same cell

Repetition detection uses a sliding window (default 4 steps). If the agent visits **≤2 unique positions** in that window, a penalty is applied.

---

## Models

### Q-Learning (default)
- Tabular Q-values per ghost
- Discrete action selection (argmax)
- Saves `ghost_<i>_qtable.json`

### MADDPG
- Actor-critic networks per ghost (PyTorch)
- Continuous action output mapped to discrete actions via `argmax`
- Saves `ghost_<i>_actor.pt` and `ghost_<i>_critic.pt`

### Coordination / Communication
- When `coordination.enabled=true`, ghosts use **intelligent encirclement** (not linear chase).
- **Strategy**: Each ghost is assigned a fixed cardinal approach direction around Pac-Man:
  - 4 ghosts → N/E/S/W at adaptive radius
  - 2 ghosts → left/right flanks
- **Adaptive radius**: shrinks as team closes in (avg_distance × 0.7, clamped 1–5), tightening the trap.
- **Wall handling**: if cardinal target is blocked, ghost falls back to direct BFS chase toward PM.
- **Capture**: when within ~2 cells, direct chase dominates for final capture.
- Coordination is always-on when enabled (both training and evaluation).
- Configure in `config.json`:

```json
"coordination": {
  "enabled": true,
  "vision_range": 8,
  "surround": true
},
"team_rewards": {
  "mobility": 1.5,
  "adjacent_block": 0.75,
  "spread": 0.5,
  "cluster_penalty": -0.6,
  "line_penalty": -1.2,
  "overlap_penalty": -0.4,
  "cluster_distance": 1
}
```

---

## Evaluation Output (Example)

```
[INFO] EVALUATION MODE
[INFO] [EVAL] Episodes: 200
[INFO] [EVAL] Test Map: Classic #1 (index 2)
[INFO] [EVAL] Ghosts: 4
...
[INFO] EVALUATION SUMMARY
[INFO] [EVAL] Ghost Wins: 120 (60.0%)
[INFO] [EVAL] Pac-Man Wins: 80 (40.0%)
[INFO] [EVAL] Avg Episode Length: 210.4 steps
```
