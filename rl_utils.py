"""
Shared utilities for RL agents in Pac-Man.
Contains common constants, state encoding, action execution, and replay buffer.
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn

# ============================================================================
# CONSTANTS
# ============================================================================
# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_NOOP = 4  # Optional

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
ACTION_NAMES = {
    ACTION_UP: "UP", 
    ACTION_DOWN: "DOWN", 
    ACTION_LEFT: "LEFT", 
    ACTION_RIGHT: "RIGHT", 
    ACTION_NOOP: "NOOP"
}

# Direction deltas
ACTION_DELTA = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
    ACTION_NOOP: (0, 0),
}

# Discretization bins for distances
DIST_BINS = [0, 2, 4, 8, 16, 30]  # 6 bins: 0-1, 2-3, 4-7, 8-15, 16-29, 30+

# Reward values
REWARD_COLLISION = 100       # All ghosts get this when ANY catches Pac-Man
REWARD_PELLETS_EATEN = -100  # All ghosts get this when Pac-Man wins
REWARD_CLOSER = 1            # Per step moving closer
REWARD_FARTHER = -1          # Per step moving away
REWARD_INVALID = -0.5        # Hit wall or invalid
REWARD_REPEAT = -0.25        # Repeated movement pattern penalty

# Repetition detection
REPEAT_PATTERN_WINDOW = 4    # Track last 4 positions
REPEAT_PATTERN_THRESHOLD = 2  # Penalize when oscillation detected

# Q-learning hyperparameters
DEFAULT_ALPHA = 0.1          # Learning rate
DEFAULT_GAMMA = 0.95         # Discount factor
DEFAULT_EPSILON = 1.0        # Initial exploration
DEFAULT_EPSILON_MIN = 0.05   # Minimum exploration
DEFAULT_EPSILON_DECAY = 0.995  # Per episode decay


# ============================================================================
# STATE ENCODING
# ============================================================================
def discretize_distance(dist: int) -> int:
    """Convert distance to bin index."""
    for i, max_val in enumerate(DIST_BINS[1:]):
        if dist < max_val:
            return i
    return len(DIST_BINS) - 1


def get_direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
    """Get 4-direction index from from_pos to to_pos."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 2  # RIGHT or LEFT
    else:
        return 1 if dy > 0 else 0  # DOWN or UP


def encode_state(ghost_pos: Tuple[int, int],
                 pacman_pos: Tuple[int, int],
                 other_ghosts: List[Tuple[int, int]],
                 walls: set,
                 pellets: List,
                 grid_size: int = 30,
                 observation_range: int = None) -> Tuple[int, ...]:
    """
    Encode game state into discrete tuple for Q-table lookup.
    
    Components:
    - dist_pm: distance to Pac-Man (6 bins, or max bin if out of observation range)
    - dir_pm: direction to Pac-Man (4 dirs, or 4=unknown if out of range)
    - dist_ghost: distance to nearest other ghost (6 bins)
    - dir_ghost: direction to nearest other ghost (4 dirs, or 4 if none)
    - can_up/down/left/right: 4 binary (can move?)
    - pellet_near: is there pellet within 3 cells? (2 bins)
    
    Args:
        observation_range: If set, Pac-Man beyond this Manhattan distance is "unknown"
    """
    # Distance and direction to Pac-Man
    dist_pm = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
    
    # Partial observability: if Pac-Man beyond observation range, encode as unknown
    if observation_range is not None and dist_pm > observation_range:
        dist_pm_bin = len(DIST_BINS) - 1  # Max/far bin
        dir_pm = 4  # Unknown direction (5th value = no direction)
    else:
        dist_pm_bin = discretize_distance(dist_pm)
        dir_pm = get_direction(ghost_pos, pacman_pos)
    
    # Nearest other ghost
    if other_ghosts:
        dists = [abs(ghost_pos[0] - g[0]) + abs(ghost_pos[1] - g[1]) for g in other_ghosts]
        nearest_idx = min(range(len(dists)), key=lambda i: dists[i])
        dist_ghost = dists[nearest_idx]
        dir_ghost = get_direction(ghost_pos, other_ghosts[nearest_idx])
    else:
        dist_ghost = grid_size * 2  # Far
        dir_ghost = 4  # No ghost
    
    dist_ghost_bin = discretize_distance(dist_ghost)
    
    # Wall proximity (can move in each direction?)
    def can_move(dx, dy):
        nx, ny = ghost_pos[0] + dx, ghost_pos[1] + dy
        return 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in walls
    
    can_up = 1 if can_move(0, -1) else 0
    can_down = 1 if can_move(0, 1) else 0
    can_left = 1 if can_move(-1, 0) else 0
    can_right = 1 if can_move(1, 0) else 0
    
    # Pellet nearby?
    pellet_near = 0
    for p in pellets:
        if getattr(p, 'active', True):
            d = abs(ghost_pos[0] - p.pos[0]) + abs(ghost_pos[1] - p.pos[1])
            if d <= 3:
                pellet_near = 1
                break
    
    return (dist_pm_bin, dir_pm, dist_ghost_bin, dir_ghost,
            can_up, can_down, can_left, can_right, pellet_near)


# ============================================================================
# ACTION EXECUTION HELPER
# ============================================================================
def execute_action(pos: Tuple[int, int], 
                   action: int, 
                   walls: set, 
                   grid_size: int = 30) -> Tuple[Tuple[int, int], bool]:
    """Execute action, return (new_pos, valid)."""
    dx, dy = ACTION_DELTA.get(action, (0, 0))
    new_pos = (pos[0] + dx, pos[1] + dy)
    
    valid = (0 <= new_pos[0] < grid_size and 
             0 <= new_pos[1] < grid_size and 
             new_pos not in walls)
    
    if not valid:
        new_pos = pos  # Stay in place
    
    return new_pos, valid


# ============================================================================
# REPLAY BUFFER (shared by MADDPG, DQN, QMIX)
# ============================================================================
class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# BASE Q-LEARNING GHOST (used as reward helper by deep models)
# ============================================================================
class QLearningGhost:
    """
    Q-learning agent for a single ghost.
    Also used as a reward computation helper by deep RL models.
    """
    
    def __init__(self, 
                 ghost_id: int,
                 alpha: float = DEFAULT_ALPHA,
                 gamma: float = DEFAULT_GAMMA,
                 epsilon: float = DEFAULT_EPSILON,
                 epsilon_min: float = DEFAULT_EPSILON_MIN,
                 epsilon_decay: float = DEFAULT_EPSILON_DECAY,
                 use_noop: bool = False,
                 repeat_window: int = REPEAT_PATTERN_WINDOW,
                 repeat_threshold: int = REPEAT_PATTERN_THRESHOLD,
                 repeat_penalty: float = REWARD_REPEAT,
                 reward_collision: float = REWARD_COLLISION,
                 reward_pellets: float = REWARD_PELLETS_EATEN,
                 reward_closer: float = REWARD_CLOSER,
                 reward_farther: float = REWARD_FARTHER,
                 reward_invalid: float = REWARD_INVALID):
        
        self.ghost_id = ghost_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_noop = use_noop
        self.repeat_window = repeat_window
        self.repeat_threshold = repeat_threshold
        self.repeat_penalty = repeat_penalty
        self.reward_collision = reward_collision
        self.reward_pellets = reward_pellets
        self.reward_closer = reward_closer
        self.reward_farther = reward_farther
        self.reward_invalid = reward_invalid
        
        # Q-table: defaultdict for sparse storage
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(len(ACTIONS) + (1 if use_noop else 0))
        )
        
        self.actions = list(ACTIONS)
        if use_noop:
            self.actions.append(ACTION_NOOP)
        
        # Tracking
        self.total_reward = 0
        self.episode_rewards = []
        self.steps = 0
        self.last_dist_to_pm = None
        self.recent_positions = deque(maxlen=self.repeat_window)
        self.repeat_penalty_count = 0
    
    def get_action(self, state: Tuple[int, ...], training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        
        q_values = self.q_table[state]
        return int(np.argmax(q_values[:len(self.actions)]))
    
    def update(self, state: Tuple[int, ...], 
               action: int, 
               reward: float, 
               next_state: Optional[Tuple[int, ...]],
               done: bool = False):
        """Q-learning update."""
        current_q = self.q_table[state][action]
        
        if done or next_state is None:
            target = reward
        else:
            max_next_q = np.max(self.q_table[next_state][:len(self.actions)])
            target = reward + self.gamma * max_next_q
        
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)
        self.total_reward += reward
        self.steps += 1
    
    def compute_reward(self, 
                       old_pos: Tuple[int, int],
                       new_pos: Tuple[int, int],
                       pacman_pos: Tuple[int, int],
                       collision: bool,
                       pellets_done: bool,
                       valid_move: bool) -> float:
        """Compute reward for a step."""
        reward = 0.0
        
        if collision:
            reward += self.reward_collision
        
        if pellets_done:
            reward += self.reward_pellets
        
        if self.last_dist_to_pm is not None:
            old_dist = self.last_dist_to_pm
            new_dist = abs(new_pos[0] - pacman_pos[0]) + abs(new_pos[1] - pacman_pos[1])
            
            if new_dist < old_dist:
                reward += self.reward_closer
            elif new_dist > old_dist:
                reward += self.reward_farther
        
        self.last_dist_to_pm = abs(new_pos[0] - pacman_pos[0]) + abs(new_pos[1] - pacman_pos[1])
        
        if not valid_move:
            reward += self.reward_invalid
        
        self.recent_positions.append(new_pos)
        if len(self.recent_positions) == self.repeat_window:
            unique_positions = len(set(self.recent_positions))
            if unique_positions <= self.repeat_threshold:
                reward += self.repeat_penalty
                self.repeat_penalty_count += 1
        
        return reward
    
    def decay_epsilon(self):
        """Decay exploration rate after episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_episode(self):
        """Reset per-episode tracking."""
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0
        self.steps = 0
        self.last_dist_to_pm = None
        self.recent_positions.clear()
        self.repeat_penalty_count = 0
    
    def save(self, filepath: str):
        """Save Q-table to JSON."""
        import json
        import os
        serializable = {
            "ghost_id": self.ghost_id,
            "params": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "q_table": {
                str(k): v.tolist() for k, v in self.q_table.items()
            },
            "stats": {
                "total_episodes": len(self.episode_rewards),
                "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
                "repeat_penalties": self.repeat_penalty_count,
            }
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def load(self, filepath: str):
        """Load Q-table from JSON."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        p = data.get("params", {})
        self.alpha = p.get("alpha", self.alpha)
        self.gamma = p.get("gamma", self.gamma)
        self.epsilon = p.get("epsilon", self.epsilon)
        
        self.q_table.clear()
        for k_str, v_list in data.get("q_table", {}).items():
            k = tuple(int(x.strip()) for x in k_str.strip("()").split(","))
            self.q_table[k] = np.array(v_list)
