"""
Q-Learning Agent for Ghost Control in Pac-Man
Each ghost is an independent Q-learning agent trying to catch Pac-Man.

State: Compact representation (discretized distances, directions)
Actions: UP, DOWN, LEFT, RIGHT (+ optional NOOP)

Rewards:
  +100  : ANY ghost collides with Pac-Man (all ghosts get this)
  -100  : Pac-Man eats all pellets (ghosts fail)
  +1    : Moved closer to Pac-Man
  -1    : Moved away from Pac-Man  
  -0.5  : Invalid action (hit wall / do nothing)
"""

import numpy as np
import json
import os
from typing import Dict, Tuple, Optional, List
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim

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
ACTION_NAMES = {ACTION_UP: "UP", ACTION_DOWN: "DOWN", ACTION_LEFT: "LEFT", 
                ACTION_RIGHT: "RIGHT", ACTION_NOOP: "NOOP"}

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
REWARD_COLLISION = 100      # All ghosts get this when ANY catches Pac-Man
REWARD_PELLETS_EATEN = -100 # All ghosts get this when Pac-Man wins
REWARD_CLOSER = 1           # Per step moving closer
REWARD_FARTHER = -1         # Per step moving away
REWARD_INVALID = -0.5       # Hit wall or invalid
REWARD_REPEAT = -0.25       # Repeated movement pattern penalty

# Repetition detection
REPEAT_PATTERN_WINDOW = 4   # Track last 4 positions
REPEAT_PATTERN_THRESHOLD = 2  # Penalize when oscillation detected

# Q-learning hyperparameters
DEFAULT_ALPHA = 0.1         # Learning rate
DEFAULT_GAMMA = 0.95        # Discount factor
DEFAULT_EPSILON = 1.0       # Initial exploration
DEFAULT_EPSILON_MIN = 0.05  # Minimum exploration
DEFAULT_EPSILON_DECAY = 0.995  # Per episode decay

# State encoding: tuple of discrete values
# (my_dist_to_pm, my_dir_to_pm, nearest_ghost_dist, nearest_ghost_dir, 
#  can_up, can_down, can_left, can_right, pellet_density_nearby)


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
                 pellets: List,  # List of Pellet objects
                 grid_size: int = 30) -> Tuple[int, ...]:
    """
    Encode game state into discrete tuple for Q-table lookup.
    
    Components:
    - dist_pm: distance to Pac-Man (6 bins)
    - dir_pm: direction to Pac-Man (4 dirs)
    - dist_ghost: distance to nearest other ghost (6 bins)
    - dir_ghost: direction to nearest other ghost (4 dirs, or 4 if none)
    - can_up/down/left/right: 4 binary (can move?)
    - pellet_near: is there pellet within 3 cells? (2 bins)
    """
    # Distance and direction to Pac-Man
    dist_pm = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
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
# Q-LEARNING AGENT
# ============================================================================
class QLearningGhost:
    """
    Q-learning agent for a single ghost.
    Each ghost has its own Q-table (or can share via class variable).
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
        # Key: state tuple, Value: array of Q-values for each action
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
        self.last_dist_to_pm = None  # For computing closer/farther reward
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
        
        # Q-update
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
        
        # Collision with Pac-Man (any ghost)
        if collision:
            reward += self.reward_collision
        
        # Pac-Man ate all pellets (ghosts fail)
        if pellets_done:
            reward += self.reward_pellets
        
        # Distance-based reward
        if self.last_dist_to_pm is not None:
            old_dist = self.last_dist_to_pm
            new_dist = abs(new_pos[0] - pacman_pos[0]) + abs(new_pos[1] - pacman_pos[1])
            
            if new_dist < old_dist:
                reward += self.reward_closer
            elif new_dist > old_dist:
                reward += self.reward_farther
        
        self.last_dist_to_pm = abs(new_pos[0] - pacman_pos[0]) + abs(new_pos[1] - pacman_pos[1])
        
        # Invalid move penalty
        if not valid_move:
            reward += self.reward_invalid
        
        # Repetition penalty (oscillation or staying in place)
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
        # Convert tuple keys to strings, numpy arrays to lists
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
        print(f"Saved Q-table for ghost {self.ghost_id} to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Restore params
        p = data.get("params", {})
        self.alpha = p.get("alpha", self.alpha)
        self.gamma = p.get("gamma", self.gamma)
        self.epsilon = p.get("epsilon", self.epsilon)
        
        # Restore Q-table
        self.q_table.clear()
        for k_str, v_list in data.get("q_table", {}).items():
            # Parse tuple key from string like "(1, 2, 0, ...)"
            k = tuple(int(x.strip()) for x in k_str.strip("()").split(","))
            self.q_table[k] = np.array(v_list)
        
        print(f"Loaded Q-table for ghost {self.ghost_id} from {filepath} ({len(self.q_table)} states)")


# ============================================================================
# MULTI-GHOST COORDINATOR (Optional: shared reward for collision)
# ============================================================================
class MultiGhostQLearning:
    """
    Manages multiple Q-learning ghosts.
    Handles shared rewards (collision bonus to all).
    """
    
    def __init__(self, num_ghosts: int = 4, **q_params):
        self.agents = [QLearningGhost(i, **q_params) for i in range(num_ghosts)]
        self.shared_collision = True  # All ghosts get collision reward
        self.repeat_window = q_params.get("repeat_window", REPEAT_PATTERN_WINDOW)
        self.repeat_threshold = q_params.get("repeat_threshold", REPEAT_PATTERN_THRESHOLD)
        self.repeat_penalty = q_params.get("repeat_penalty", REWARD_REPEAT)
        self.reward_collision = q_params.get("reward_collision", REWARD_COLLISION)
        self.reward_pellets = q_params.get("reward_pellets", REWARD_PELLETS_EATEN)
        self.reward_closer = q_params.get("reward_closer", REWARD_CLOSER)
        self.reward_farther = q_params.get("reward_farther", REWARD_FARTHER)
        self.reward_invalid = q_params.get("reward_invalid", REWARD_INVALID)
    
    def get_actions(self, states: List[Tuple[int, ...]], training: bool = True) -> List[int]:
        """Get actions for all ghosts."""
        return [agent.get_action(s, training) for agent, s in zip(self.agents, states)]
    
    def update_all(self, 
                   states: List[Tuple[int, ...]],
                   actions: List[int],
                   rewards: List[float],
                   next_states: List[Optional[Tuple[int, ...]]],
                   done: bool = False):
        """Update all agents."""
        for agent, s, a, r, ns in zip(self.agents, states, actions, rewards, next_states):
            agent.update(s, a, r, ns, done)
    
    def compute_rewards(self,
                        old_positions: List[Tuple[int, int]],
                        new_positions: List[Tuple[int, int]],
                        pacman_pos: Tuple[int, int],
                        collision: bool,
                        pellets_done: bool,
                        valid_moves: List[bool]) -> List[float]:
        """Compute rewards for all ghosts, handling shared collision."""
        rewards = []
        for i, (old_pos, new_pos, valid) in enumerate(zip(old_positions, new_positions, valid_moves)):
            agent = self.agents[i]
            r = agent.compute_reward(old_pos, new_pos, pacman_pos, 
                                     collision, pellets_done, valid)
            rewards.append(r)
        
        # Share collision reward across all ghosts
        if collision and self.shared_collision:
            for i in range(len(rewards)):
                rewards[i] = max(rewards[i], REWARD_COLLISION)
        
        return rewards
    
    def decay_all(self):
        """Decay epsilon for all agents."""
        for agent in self.agents:
            agent.decay_epsilon()
    
    def reset_all(self):
        """Reset all agents for new episode."""
        for agent in self.agents:
            agent.reset_episode()
    
    def save_all(self, directory: str):
        """Save all Q-tables."""
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(directory, f"ghost_{i}_qtable.json"))
    
    def load_all(self, directory: str):
        """Load all Q-tables."""
        for i, agent in enumerate(self.agents):
            path = os.path.join(directory, f"ghost_{i}_qtable.json")
            if os.path.exists(path):
                agent.load(path)
            else:
                print(f"No Q-table found for ghost {i} at {path}")


# ============================================================================
# MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
# ============================================================================
class Actor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        last = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            last = size
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x))


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        last = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            last = size
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
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


class MADDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int],
                 actor_lr: float, critic_lr: float, gamma: float, tau: float,
                 epsilon_start: float, epsilon_min: float, epsilon_decay: float):
        self.actor = Actor(state_dim, action_dim, hidden_sizes)
        self.actor_target = Actor(state_dim, action_dim, hidden_sizes)
        self.critic = Critic(state_dim + action_dim, hidden_sizes)
        self.critic_target = Critic(state_dim + action_dim, hidden_sizes)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_reward = 0
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, training: bool = True):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_t).detach().cpu().numpy()[0]
        if training:
            noise = self.epsilon * np.random.randn(*action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        self.total_reward += float(np.mean(rewards))
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(torch.cat([next_states_t, next_actions], dim=1))
            y = rewards_t + self.gamma * (1 - dones_t) * target_q
        
        current_q = self.critic(torch.cat([states_t, actions_t], dim=1))
        critic_loss = nn.MSELoss()(current_q, y)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # Actor update
        actor_loss = -self.critic(torch.cat([states_t, self.actor(states_t)], dim=1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # Soft update targets
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class MultiGhostMADDPG:
    """Multi-agent MADDPG wrapper for ghosts."""
    def __init__(self, num_ghosts: int, state_dim: int, action_dim: int, config: dict,
                 repeat_window: int = REPEAT_PATTERN_WINDOW,
                 repeat_threshold: int = REPEAT_PATTERN_THRESHOLD,
                 repeat_penalty: float = REWARD_REPEAT):
        maddpg_cfg = config.get("maddpg", {})
        hidden_sizes = maddpg_cfg.get("hidden_sizes", [128, 128])
        
        self.agents = [
            MADDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                actor_lr=maddpg_cfg.get("actor_lr", 0.0005),
                critic_lr=maddpg_cfg.get("critic_lr", 0.001),
                gamma=maddpg_cfg.get("gamma", 0.95),
                tau=maddpg_cfg.get("tau", 0.01),
                epsilon_start=maddpg_cfg.get("epsilon_start", 0.2),
                epsilon_min=maddpg_cfg.get("epsilon_min", 0.02),
                epsilon_decay=maddpg_cfg.get("epsilon_decay", 0.995),
            )
            for _ in range(num_ghosts)
        ]
        self.num_ghosts = num_ghosts
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = ReplayBuffer(maddpg_cfg.get("buffer_size", 100000))
        self.batch_size = maddpg_cfg.get("batch_size", 128)
        self.update_every = maddpg_cfg.get("update_every", 1)
        self.timestep = 0
        self.shared_collision = True
        self.last_action_cont = [np.zeros(action_dim, dtype=np.float32) for _ in range(num_ghosts)]
        
        # Reward helpers to reuse Q-learning shaping logic
        self.reward_helpers = [
            QLearningGhost(
                ghost_id=i,
                repeat_window=repeat_window,
                repeat_threshold=repeat_threshold,
                repeat_penalty=repeat_penalty
            )
            for i in range(num_ghosts)
        ]
    
    def get_actions(self, states: List[Tuple[int, ...]], training: bool = True) -> List[int]:
        actions = []
        for i, (agent, state) in enumerate(zip(self.agents, states)):
            action_cont = agent.select_action(np.array(state, dtype=np.float32), training=training)
            self.last_action_cont[i] = action_cont
            action_idx = int(np.argmax(action_cont))
            actions.append(action_idx)
        return actions

    def update_all(self, states, actions, rewards, next_states, done):
        for i, agent in enumerate(self.agents):
            self.buffer.push(states[i], self.last_action_cont[i], rewards[i], next_states[i], done)
        
        self.timestep += 1
        if len(self.buffer) >= self.batch_size and self.timestep % self.update_every == 0:
            batch = self.buffer.sample(self.batch_size)
            for agent in self.agents:
                agent.update(batch)
        
        if done:
            for agent in self.agents:
                agent.decay_epsilon()

    def compute_rewards(self,
                        old_positions: List[Tuple[int, int]],
                        new_positions: List[Tuple[int, int]],
                        pacman_pos: Tuple[int, int],
                        collision: bool,
                        pellets_done: bool,
                        valid_moves: List[bool]) -> List[float]:
        rewards = []
        for i, (old_pos, new_pos, valid) in enumerate(zip(old_positions, new_positions, valid_moves)):
            helper = self.reward_helpers[i]
            r = helper.compute_reward(old_pos, new_pos, pacman_pos, collision, pellets_done, valid)
            rewards.append(r)
        
        if collision and self.shared_collision:
            for i in range(len(rewards)):
                rewards[i] = max(rewards[i], REWARD_COLLISION)
        
        return rewards

    def reset_all(self):
        for helper in self.reward_helpers:
            helper.reset_episode()
        for agent in self.agents:
            agent.total_reward = 0
        self.last_action_cont = [np.zeros(self.action_dim, dtype=np.float32) for _ in range(self.num_ghosts)]

    def save_all(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(directory, f"ghost_{i}_actor.pt"))
            torch.save(agent.critic.state_dict(), os.path.join(directory, f"ghost_{i}_critic.pt"))

    def load_all(self, directory: str):
        for i, agent in enumerate(self.agents):
            actor_path = os.path.join(directory, f"ghost_{i}_actor.pt")
            critic_path = os.path.join(directory, f"ghost_{i}_critic.pt")
            if os.path.exists(actor_path):
                agent.actor.load_state_dict(torch.load(actor_path))
            if os.path.exists(critic_path):
                agent.critic.load_state_dict(torch.load(critic_path))



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
