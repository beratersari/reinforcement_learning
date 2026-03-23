"""
Ghost Roles System - Multi-Head Policy for Pac-Man Ghosts.

Each ghost can take on one of three roles:
- CHASER: Directly pursues Pac-Man
- BLOCKER: Blocks Pac-Man's escape routes
- AMBUSHER: Waits ahead of Pac-Man's predicted path

Ghosts learn to select roles via a role head, enabling emergent specialization.
"""

import numpy as np
from enum import IntEnum
from typing import List


class GhostRole(IntEnum):
    """Role enum for ghost specialization."""
    CHASER = 0
    BLOCKER = 1
    AMBUSHER = 2
    
    @classmethod
    def names(cls):
        return ["Chaser", "Blocker", "Ambusher"]
    
    @classmethod
    def count(cls):
        return 3


class RoleManager:
    """
    Manages role selection and role-aware behavior for ghosts.
    
    Each ghost has a role that influences its action selection.
    Role selection can be:
    - Learned (via role Q-head for neural models)
    - Random with exploration
    """
    
    def __init__(self, num_ghosts: int, epsilon_start: float = 1.0, 
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995):
        self.num_ghosts = num_ghosts
        # Initialize with diverse roles based on ghost count
        self.roles = self._initialize_diverse_roles(num_ghosts)
        self.role_epsilon = epsilon_start
        self.role_epsilon_min = epsilon_min
        self.role_epsilon_decay = epsilon_decay
        self.role_q_tables = {}  # For Q-learning role selection
    
    def _initialize_diverse_roles(self, num_ghosts: int) -> List[GhostRole]:
        """Initialize ghosts with diverse roles based on count."""
        all_roles = list(GhostRole)
        roles = []
        for i in range(num_ghosts):
            # Cycle through available roles for diversity
            roles.append(all_roles[i % len(all_roles)])
        return roles
    
    def get_role_name(self, role: GhostRole) -> str:
        """Get display name for role."""
        return GhostRole.names()[role]
    
    def get_role_onehot(self, role: GhostRole) -> np.ndarray:
        """Get one-hot encoding of role."""
        onehot = np.zeros(GhostRole.count())
        onehot[role] = 1.0
        return onehot
    
    def select_role(self, ghost_idx: int, training: bool = True) -> GhostRole:
        """
        Select role for a ghost.
        
        During training: epsilon-greedy over roles
        During eval: greedy (use current role)
        """
        if training and np.random.random() < self.role_epsilon:
            role = GhostRole(np.random.randint(GhostRole.count()))
        else:
            role = self.roles[ghost_idx]
        return role
    
    def update_role(self, ghost_idx: int, new_role: GhostRole):
        """Update ghost's current role."""
        self.roles[ghost_idx] = new_role
    
    def decay_role_epsilon(self):
        """Decay role exploration epsilon."""
        self.role_epsilon = max(self.role_epsilon_min, 
                                self.role_epsilon * self.role_epsilon_decay)
    
    def get_role_embedding(self, ghost_idx: int) -> np.ndarray:
        """Get role embedding for state augmentation."""
        return self.get_role_onehot(self.roles[ghost_idx])
    
    def get_all_role_embeddings(self) -> np.ndarray:
        """Get all role embeddings concatenated."""
        return np.concatenate([self.get_role_onehot(r) for r in self.roles])


# Role-specific action modifiers (for heuristic guidance)
# These can be used as action masks or reward bonuses
ROLE_PREFERENCES = {
    GhostRole.CHASER: {
        "prefer_toward_pm": 1.0,
        "prefer_block": 0.3,
        "prefer_ambush": 0.1
    },
    GhostRole.BLOCKER: {
        "prefer_toward_pm": 0.3,
        "prefer_block": 1.0,
        "prefer_ambush": 0.5
    },
    GhostRole.AMBUSHER: {
        "prefer_toward_pm": 0.2,
        "prefer_block": 0.4,
        "prefer_ambush": 1.0
    }
}
