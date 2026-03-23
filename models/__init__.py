"""
RL Models for Ghost Control in Pac-Man.

Available models:
- MultiGhostQLearning: Tabular Q-learning with independent Q-tables
- MultiGhostMADDPG: Multi-Agent DDPG with actor-critic
- MultiGhostDQN: Shared Deep Q-Network
- MultiGhostPPO: Proximal Policy Optimization
- MultiGhostQMIX: Factorized Multi-Agent Q-Learning
- MultiGhostVDN: Value Decomposition Networks

Roles: Each ghost picks a role (Chaser/Blocker/Ambusher) via multi-head policy.
"""

from .qlearning import MultiGhostQLearning
from .maddpg import MultiGhostMADDPG
from .dqn import MultiGhostDQN
from .ppo import MultiGhostPPO
from .qmix import MultiGhostQMIX
from .vdn import MultiGhostVDN

# Optional roles import (for multi-head policy)
try:
    from .roles import GhostRole, RoleManager
    _roles_available = True
except ImportError:
    _roles_available = False
    GhostRole = None
    RoleManager = None

__all__ = [
    'MultiGhostQLearning',
    'MultiGhostMADDPG',
    'MultiGhostDQN',
    'MultiGhostPPO',
    'MultiGhostQMIX',
    'MultiGhostVDN',
    'GhostRole',
    'RoleManager',
]
