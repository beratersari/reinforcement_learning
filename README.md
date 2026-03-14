# Pac-Man RL Simulation

A 30x30 Pac-Man simulation environment for training reinforcement learning agents to control ghosts.

## Features

- **Game Simulation**
  - 30x30 grid with auto-generated connected mazes (no unreachable areas)
  - Multiple map types: Random, Recursive Maze, Classic-style
  - Map selection menu at startup
  - Pac-Man uses heuristic AI (BFS pathfinding + ghost escape)
  - 4 colorful ghosts (Red, Cyan, Orange, Pink) that block each other

- **Q-Learning for Ghosts** (New!)
  - Each ghost is an independent Q-learning agent
  - Configurable ghost count (2 or 4 ghosts)
  - Compact state encoding (distances, directions, wall proximity)
  - Shared collision reward (+100 when ANY ghost catches Pac-Man)
  - Penalty for Pac-Man eating all pellets (-100)
  - Distance-based shaping (+1 closer, -1 farther)
  - Invalid move penalty (-0.5)
  - Epsilon-greedy exploration with decay
  - Save/load Q-tables for transfer learning
  - **Watch training live** with `--render` or `--render-every` flags

## Installation

```bash
pip install -r requirements.txt
```

## Running the Game (Watch Mode)

```bash
python pacman_game.py
```

**Controls:**
- **Menu**: `1-9` select map | `UP/DOWN` navigate | `ENTER` start
- **Game**: `R` restart | `N` next map | `M` menu | `ESC` quit

## Training Ghosts with Q-Learning

```bash
# Train from scratch (default: 4 ghosts)
python train_ghosts.py --episodes 2000 --save-dir models/

# Train with 2 ghosts (compare performance!)
python train_ghosts.py --ghosts 2 --episodes 1000 --save-dir models_2ghosts/

# Continue training from saved models
python train_ghosts.py --load-dir models/ --episodes 1000

# Watch training live (every episode)
python train_ghosts.py --render --episodes 100 --fps 10

# Watch every 10th episode (faster)
python train_ghosts.py --render-every 10 --episodes 500

# Quick test (500 episodes)
python train_ghosts.py --episodes 500 --save-every 50
```

**Training output:**
```
Ep    50 | AvgR:   -234.2 | GhostWin:  12.0% | ε: 0.778
Ep   100 | AvgR:     45.3 | GhostWin:  28.0% | ε: 0.605
...
✓ Training complete! Models saved to models/
  Final ghost win rate: 67.3%
```

### Comparing 2 vs 4 Ghosts

Run experiments to compare performance:

```bash
# Experiment 1: 2 ghosts
python train_ghosts.py --ghosts 2 --episodes 1000 --save-dir exp_2ghosts/

# Experiment 2: 4 ghosts
python train_ghosts.py --ghosts 4 --episodes 1000 --save-dir exp_4ghosts/
```

Expected results:
- **2 ghosts**: Lower win rate initially, but faster learning per ghost. Better for studying coordination.
- **4 ghosts**: Higher win rate due to numbers, but more complex coordination needed.

## Project Structure

```
.
├── pacman_game.py      # Main game + map generator + visualization
├── ghost_rl.py         # Q-learning agents (QLearningGhost, MultiGhostQLearning)
├── train_ghosts.py     # Training script (headless, CLI)
├── requirements.txt    # Dependencies
├── README.md           # This file
└── models/             # Saved Q-tables (created after training)
    ├── ghost_0_qtable.json
    ├── ghost_1_qtable.json
    └── ...
```

## State Representation

Each ghost observes a compact 9-element state tuple:

| Index | Description | Bins |
|-------|-------------|------|
| 0 | Distance to Pac-Man | 6 (0-1, 2-3, 4-7, 8-15, 16-29, 30+) |
| 1 | Direction to Pac-Man | 4 (UP/DOWN/LEFT/RIGHT) |
| 2 | Distance to nearest ghost | 6 |
| 3 | Direction to nearest ghost | 5 (4 dirs + none) |
| 4-7 | Can move (UP/DOWN/LEFT/RIGHT)? | 2 each (binary) |
| 8 | Pellet nearby (≤3 cells)? | 2 (binary) |

**Total state space**: ~10,000 unique states (sparse via defaultdict)

## Rewards

| Event | Reward |
|-------|--------|
| ANY ghost catches Pac-Man | +100 (all ghosts) |
| Pac-Man eats all pellets | -100 (all ghosts) |
| Moved closer to Pac-Man | +1 |
| Moved away from Pac-Man | -1 |
| Invalid move (wall) | -0.5 |

## Hyperparameters

| Param | Default | Description |
|-------|---------|-------------|
| α (alpha) | 0.1 | Learning rate |
| γ (gamma) | 0.95 | Discount factor |
| ε start | 1.0 | Initial exploration |
| ε min | 0.05 | Minimum exploration |
| ε decay | 0.995 | Per-episode decay |
| Ghosts | 4 | Number of ghosts (2 or 4) |
| FPS | 10 | Rendering speed (when watching) |

### CLI Arguments

| Flag | Description |
|------|-------------|
| `--episodes N` | Number of training episodes |
| `--ghosts N` | Number of ghosts (2 or 4) |
| `--pellets N` | Max pellets (default 100) |
| `--time-limit N` | Seconds for Pac-Man timeout (default 20) |
| `--save-dir PATH` | Where to save Q-tables |
| `--load-dir PATH` | Load existing models |
| `--save-every N` | Save checkpoint every N episodes |
| `--render` | Show game window (all episodes) |
| `--render-every N` | Show every Nth episode |
| `--fps N` | Rendering speed (default 10) |

### Win Conditions

Pac-Man wins if:
- **Collects all pellets**, OR  
- **Survives until timer expires** (default 20s)

Timer shown on GUI as countdown ⏱. Color changes: white → yellow → red.

**Time penalty for ghosts**: -100 if time expires (same as all pellets eaten).

## Future Work

- PPO / A3C / DQN for better sample efficiency
- Shared Q-network across ghosts (parameter sharing)
- Curriculum learning (start with simpler maps)
- Train Pac-Man as well (competitive multi-agent)

## License

MIT