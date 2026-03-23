"""
VDN (Value Decomposition Networks) for Ghost Control.

VDN is a simpler value decomposition method than QMIX. Instead of using a
mixing network to combine individual Q-values, VDN simply sums them:

Q_total = sum(Q_i) for all agents i

This assumes that the joint Q-value can be decomposed as a sum of individual
Q-values, which works well when agents have independent contributions to
the team's reward.
"""

import os
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rl_utils import QLearningGhost, ReplayBuffer, REWARD_COLLISION, REPEAT_PATTERN_WINDOW, REPEAT_PATTERN_THRESHOLD, REWARD_REPEAT


class QNetwork(nn.Module):
    """Individual Q-network for each ghost."""
    def __init__(self, input_dim, action_dim, hidden_sizes):
        super().__init__()
        layers = []
        last = input_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(last, size), nn.ReLU()])
            last = size
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VDNAgent:
    """Individual VDN agent for a single ghost."""
    def __init__(self, ghost_id, state_dim, action_dim, hidden_sizes, lr, gamma, tau):
        self.ghost_id = ghost_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.q_network = QNetwork(state_dim, action_dim, hidden_sizes)
        self.target_network = QNetwork(state_dim, action_dim, hidden_sizes)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.total_reward = 0
        self.episode_rewards = []

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_dim))
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def soft_update_target(self):
        """Soft update target network parameters."""
        for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def decay_epsilon(self):
        """Decay epsilon and store episode rewards."""
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        """Reset for new episode."""
        self.total_reward = 0


class MultiGhostVDN:
    """
    VDN (Value Decomposition Networks) for multi-ghost control.
    
    VDN decomposes the joint Q-value as a sum of individual Q-values:
    Q_total = sum(Q_i) for all agents i
    
    This is simpler than QMIX (no mixing network) but still allows for
    centralized training with decentralized execution.
    """
    
    def __init__(self, num_ghosts, state_dim, action_dim, config, 
                 repeat_window=REPEAT_PATTERN_WINDOW, 
                 repeat_threshold=REPEAT_PATTERN_THRESHOLD, 
                 repeat_penalty=REWARD_REPEAT):
        cfg = config.get("vdn", {})
        hidden = cfg.get("hidden_sizes", [128, 128])
        
        self.num_ghosts = num_ghosts
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.get("gamma", 0.95)
        self.tau = cfg.get("tau", 0.005)
        self.batch_size = cfg.get("batch_size", 128)
        self.buffer_size = cfg.get("buffer_size", 100000)
        self.update_every = cfg.get("update_every", 4)
        self.lr = cfg.get("lr", 0.0005)
        
        # Create individual agents
        self.agents = [VDNAgent(i, state_dim, action_dim, hidden, self.lr, self.gamma, self.tau) 
                       for i in range(num_ghosts)]
        
        # Shared replay buffer
        self.buffer = ReplayBuffer(self.buffer_size)
        self.timestep = 0
        self.shared_collision = True
        
        # Reward helpers for each ghost
        self.reward_helpers = [QLearningGhost(ghost_id=i, 
                                                repeat_window=repeat_window, 
                                                repeat_threshold=repeat_threshold, 
                                                repeat_penalty=repeat_penalty) 
                               for i in range(num_ghosts)]
        
        # Epsilon parameters
        self.epsilon_start = cfg.get("epsilon_start", 1.0)
        self.epsilon_min = cfg.get("epsilon_min", 0.05)
        self.epsilon_decay = cfg.get("epsilon_decay", 0.995)
        
        for agent in self.agents:
            agent.epsilon = self.epsilon_start
            agent.epsilon_min = self.epsilon_min
            agent.epsilon_decay = self.epsilon_decay

    def get_actions(self, states, training=True):
        """Get actions for all ghosts."""
        return [agent.select_action(np.array(state, dtype=np.float32), training=training) 
                for agent, state in zip(self.agents, states)]

    def update_all(self, states, actions, rewards, next_states, done):
        """Store transition and update networks if ready."""
        transition = {
            "states": states, 
            "actions": actions, 
            "rewards": rewards, 
            "next_states": next_states, 
            "done": done
        }
        self.buffer.push(transition, actions[0], sum(rewards), next_states, done)
        
        for i, agent in enumerate(self.agents):
            agent.total_reward += rewards[i]
        
        self.timestep += 1
        if len(self.buffer) >= self.batch_size and self.timestep % self.update_every == 0:
            self._learn()
        
        if done:
            for agent in self.agents:
                agent.decay_epsilon()
            avg_eps = np.mean([a.epsilon for a in self.agents])
            for agent in self.agents:
                agent.epsilon = avg_eps

    def _learn(self):
        """Update Q-networks using VDN loss."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer.buffer[i] for i in idx]
        
        states_batch = [b[0]["states"] for b in batch]
        actions_batch = [b[0]["actions"] for b in batch]
        rewards_batch = [b[0]["rewards"] for b in batch]
        next_states_batch = [b[0]["next_states"] for b in batch]
        dones_batch = [b[0]["done"] for b in batch]
        
        # Compute VDN loss
        total_loss = 0
        for i in range(len(states_batch)):
            states = states_batch[i]
            actions = actions_batch[i]
            rewards = rewards_batch[i]
            next_states = next_states_batch[i]
            done = dones_batch[i]
            
            # Current Q-values: sum of individual Q-values for taken actions
            current_q_total = torch.zeros(1)
            for j in range(self.num_ghosts):
                state_t = torch.FloatTensor(np.array(states[j], dtype=np.float32)).unsqueeze(0)
                q_values = self.agents[j].q_network(state_t)
                current_q_total = current_q_total + q_values[0, actions[j]]
            
            # Next Q-values: sum of max individual Q-values
            with torch.no_grad():
                next_q_total = torch.zeros(1)
                for j in range(self.num_ghosts):
                    next_state_t = torch.FloatTensor(np.array(next_states[j], dtype=np.float32)).unsqueeze(0)
                    next_q = self.agents[j].target_network(next_state_t)
                    next_q_total = next_q_total + torch.max(next_q, dim=1)[0]
                
                target_q = torch.FloatTensor([sum(rewards)]).unsqueeze(0) + \
                           self.gamma * next_q_total.unsqueeze(0) * (1 - torch.FloatTensor([done]).unsqueeze(0))
            
            loss = nn.MSELoss()(current_q_total.unsqueeze(0), target_q)
            total_loss = total_loss + loss
        
        avg_loss = total_loss / len(states_batch)
        
        # Optimize all agents
        for agent in self.agents:
            agent.optimizer.zero_grad()
        avg_loss.backward()
        for agent in self.agents:
            agent.optimizer.step()
        
        # Soft update target networks
        for agent in self.agents:
            agent.soft_update_target()

    def compute_rewards(self, old_positions, new_positions, pacman_pos, collision, pellets_done, valid_moves):
        """Compute rewards for all ghosts."""
        rewards = []
        for i, (old_pos, new_pos, valid) in enumerate(zip(old_positions, new_positions, valid_moves)):
            r = self.reward_helpers[i].compute_reward(old_pos, new_pos, pacman_pos, collision, pellets_done, valid)
            rewards.append(r)
        if collision and self.shared_collision:
            for i in range(len(rewards)):
                rewards[i] = max(rewards[i], REWARD_COLLISION)
        return rewards

    def reset_all(self):
        """Reset all agents for new episode."""
        for h in self.reward_helpers:
            h.reset_episode()
        for agent in self.agents:
            agent.reset_episode()

    def get_epsilon(self):
        """Get average epsilon across all agents."""
        return float(np.mean([a.epsilon for a in self.agents])) if self.agents else 0.0

    def uses_epsilon(self):
        """Check if model uses epsilon exploration."""
        return True

    def save_all(self, directory):
        """Save all agent networks."""
        os.makedirs(directory, exist_ok=True)
        for i, agent in enumerate(self.agents):
            torch.save(agent.q_network.state_dict(), 
                       os.path.join(directory, f"ghost_{i}_vdn_q.pt"))

    def load_all(self, directory):
        """Load all agent networks."""
        for i, agent in enumerate(self.agents):
            path = os.path.join(directory, f"ghost_{i}_vdn_q.pt")
            if os.path.exists(path):
                agent.q_network.load_state_dict(torch.load(path))
                agent.target_network.load_state_dict(torch.load(path))
