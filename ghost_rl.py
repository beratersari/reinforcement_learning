"""
Backwards compatibility module for RL agents.
All implementations have been moved to separate modules.

This module re-exports everything for backwards compatibility.
New code should import directly from the specific modules:
- rl_utils: Shared utilities (constants, state encoding, replay buffer)
- models.qlearning: Q-Learning agent
- models.maddpg: MADDPG agent
- models.dqn: DQN agent
- models.ppo: PPO agent
- models.qmix: QMIX agent
"""

# Re-export everything from the new modules for backwards compatibility
from rl_utils import (
    # Constants
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_NOOP,
    ACTIONS, ACTION_NAMES, ACTION_DELTA,
    DIST_BINS,
    REWARD_COLLISION, REWARD_PELLETS_EATEN, REWARD_CLOSER, REWARD_FARTHER,
    REWARD_INVALID, REWARD_REPEAT,
    REPEAT_PATTERN_WINDOW, REPEAT_PATTERN_THRESHOLD,
    DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_EPSILON, DEFAULT_EPSILON_MIN,
    DEFAULT_EPSILON_DECAY,
    # Functions
    discretize_distance, get_direction, encode_state, execute_action,
    # Classes
    ReplayBuffer, QLearningGhost,
)

from models import (
    MultiGhostQLearning,
    MultiGhostMADDPG,
    MultiGhostDQN,
    MultiGhostPPO,
    MultiGhostQMIX,
)

__all__ = [
    # Constants
    'ACTION_UP', 'ACTION_DOWN', 'ACTION_LEFT', 'ACTION_RIGHT', 'ACTION_NOOP',
    'ACTIONS', 'ACTION_NAMES', 'ACTION_DELTA',
    'DIST_BINS',
    'REWARD_COLLISION', 'REWARD_PELLETS_EATEN', 'REWARD_CLOSER', 'REWARD_FARTHER',
    'REWARD_INVALID', 'REWARD_REPEAT',
    'REPEAT_PATTERN_WINDOW', 'REPEAT_PATTERN_THRESHOLD',
    'DEFAULT_ALPHA', 'DEFAULT_GAMMA', 'DEFAULT_EPSILON', 'DEFAULT_EPSILON_MIN',
    'DEFAULT_EPSILON_DECAY',
    # Functions
    'discretize_distance', 'get_direction', 'encode_state', 'execute_action',
    # Classes
    'ReplayBuffer', 'QLearningGhost',
    'MultiGhostQLearning', 'MultiGhostMADDPG', 'MultiGhostDQN',
    'MultiGhostPPO', 'MultiGhostQMIX',
]
