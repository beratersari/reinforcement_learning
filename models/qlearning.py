"""
Q-Learning Agent for Ghost Control in Pac-Man.
Tabular Q-learning with independent Q-tables per ghost.
"""

import os
from typing import List, Tuple

from rl_utils import (
    QLearningGhost,
    REWARD_COLLISION,
    REPEAT_PATTERN_WINDOW,
    REPEAT_PATTERN_THRESHOLD,
    REWARD_REPEAT,
)


class MultiGhostQLearning:
    """
    Manages multiple Q-learning ghosts.
    Handles shared rewards (collision bonus to all).
    """
    
    def __init__(self, num_ghosts: int = 4, **q_params):
        self.agents = [QLearningGhost(i, **q_params) for i in range(num_ghosts)]
        self.shared_collision = True
        self.repeat_window = q_params.get("repeat_window", REPEAT_PATTERN_WINDOW)
        self.repeat_threshold = q_params.get("repeat_threshold", REPEAT_PATTERN_THRESHOLD)
        self.repeat_penalty = q_params.get("repeat_penalty", REWARD_REPEAT)
        self.reward_collision = q_params.get("reward_collision", REWARD_COLLISION)
    
    def get_actions(self, states: List[Tuple[int, ...]], training: bool = True) -> List[int]:
        return [agent.get_action(s, training) for agent, s in zip(self.agents, states)]
    
    def update_all(self, states, actions, rewards, next_states, done: bool = False):
        for agent, s, a, r, ns in zip(self.agents, states, actions, rewards, next_states):
            agent.update(s, a, r, ns, done)
    
    def compute_rewards(self, old_positions, new_positions, pacman_pos, collision, pellets_done, valid_moves):
        rewards = []
        for i, (old_pos, new_pos, valid) in enumerate(zip(old_positions, new_positions, valid_moves)):
            agent = self.agents[i]
            r = agent.compute_reward(old_pos, new_pos, pacman_pos, collision, pellets_done, valid)
            rewards.append(r)
        if collision and self.shared_collision:
            for i in range(len(rewards)):
                rewards[i] = max(rewards[i], REWARD_COLLISION)
        return rewards
    
    def decay_all(self):
        for agent in self.agents:
            agent.decay_epsilon()
    
    def reset_all(self):
        for agent in self.agents:
            agent.reset_episode()
    
    def save_all(self, directory: str):
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(directory, f"ghost_{i}_qtable.json"))
    
    def load_all(self, directory: str):
        for i, agent in enumerate(self.agents):
            path = os.path.join(directory, f"ghost_{i}_qtable.json")
            if os.path.exists(path):
                agent.load(path)
    
    def get_epsilon(self) -> float:
        import numpy as np
        if not self.agents:
            return 0.0
        return float(np.mean([agent.epsilon for agent in self.agents]))
    
    def uses_epsilon(self) -> bool:
        return True
