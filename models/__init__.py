"""
RL Models for Ghost Control in Pac-Man.

Available models:
- MultiGhostQLearning: Tabular Q-learning with independent Q-tables
- MultiGhostMADDPG: Multi-Agent DDPG with actor-critic
- MultiGhostDQN: Shared Deep Q-Network
- MultiGhostPPO: Proximal Policy Optimization
- MultiGhostQMIX: Factorized Multi-Agent Q-Learning
"""

from .qlearning import MultiGhostQLearning
from .maddpg import MultiGhostMADDPG
from .dqn import MultiGhostDQN
from .ppo import MultiGhostPPO
from .qmix import MultiGhostQMIX

__all__ = [
    'MultiGhostQLearning',
    'MultiGhostMADDPG',
    'MultiGhostDQN',
    'MultiGhostPPO',
    'MultiGhostQMIX',
]
