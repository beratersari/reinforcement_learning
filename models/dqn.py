"""
Shared DQN (Deep Q-Network) for Ghost Control.
"""

import os
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rl_utils import QLearningGhost, ReplayBuffer, REWARD_COLLISION, REPEAT_PATTERN_WINDOW, REPEAT_PATTERN_THRESHOLD, REWARD_REPEAT


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        last = input_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(last, size), nn.ReLU()])
            last = size
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SharedDQN:
    def __init__(self, state_dim, action_dim, hidden_sizes, lr, gamma, buffer_size, batch_size, update_every, tau):
        self.q_network = QNetwork(state_dim, action_dim, hidden_sizes)
        self.target_network = QNetwork(state_dim, action_dim, hidden_sizes)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        self.timestep = 0
        self.action_dim = action_dim

    def select_action(self, state, epsilon, training=True):
        if training and np.random.random() < epsilon:
            return int(np.random.choice(self.action_dim))
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def step(self):
        self.timestep += 1
        if len(self.buffer) >= self.batch_size and self.timestep % self.update_every == 0:
            self._learn()

    def _learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        q_expected = self.q_network(states_t).gather(1, actions_t)
        with torch.no_grad():
            q_next = self.target_network(next_states_t).max(1)[0].unsqueeze(1)
            q_target = rewards_t + (self.gamma * q_next * (1 - dones_t))
        loss = self.criterion(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


class DQNGhostAgent:
    def __init__(self, ghost_id, epsilon_start, epsilon_min, epsilon_decay):
        self.ghost_id = ghost_id
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_reward = 0
        self.episode_rewards = []

    def decay_epsilon(self):
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        self.total_reward = 0


class MultiGhostDQN:
    def __init__(self, num_ghosts, state_dim, action_dim, config, repeat_window=REPEAT_PATTERN_WINDOW, repeat_threshold=REPEAT_PATTERN_THRESHOLD, repeat_penalty=REWARD_REPEAT):
        cfg = config.get("dqn", {})
        hidden = cfg.get("hidden_sizes", [128, 128])
        self.shared_dqn = SharedDQN(state_dim, action_dim, hidden, cfg.get("lr", 0.0005), cfg.get("gamma", 0.95), cfg.get("buffer_size", 100000), cfg.get("batch_size", 128), cfg.get("update_every", 4), cfg.get("tau", 0.001))
        self.agents = [DQNGhostAgent(i, cfg.get("epsilon_start", 1.0), cfg.get("epsilon_min", 0.05), cfg.get("epsilon_decay", 0.995)) for i in range(num_ghosts)]
        self.num_ghosts = num_ghosts
        self.shared_collision = True
        self.reward_helpers = [QLearningGhost(ghost_id=i, repeat_window=repeat_window, repeat_threshold=repeat_threshold, repeat_penalty=repeat_penalty) for i in range(num_ghosts)]

    def _augment_state(self, state, ghost_id):
        one_hot = np.zeros(self.num_ghosts, dtype=np.float32)
        if 0 <= ghost_id < self.num_ghosts:
            one_hot[ghost_id] = 1.0
        return np.concatenate([np.array(state, dtype=np.float32), one_hot])

    def get_actions(self, states, training=True):
        actions = []
        for agent, state in zip(self.agents, states):
            augmented = self._augment_state(state, agent.ghost_id)
            action = self.shared_dqn.select_action(augmented, agent.epsilon, training=training)
            actions.append(action)
        return actions

    def update_all(self, states, actions, rewards, next_states, done):
        for i, agent in enumerate(self.agents):
            self.shared_dqn.store_transition(self._augment_state(states[i], agent.ghost_id), actions[i], rewards[i], self._augment_state(next_states[i], agent.ghost_id), done)
            agent.total_reward += rewards[i]
        self.shared_dqn.step()
        if done:
            for agent in self.agents:
                agent.decay_epsilon()

    def compute_rewards(self, old_positions, new_positions, pacman_pos, collision, pellets_done, valid_moves):
        rewards = []
        for i, (old_pos, new_pos, valid) in enumerate(zip(old_positions, new_positions, valid_moves)):
            r = self.reward_helpers[i].compute_reward(old_pos, new_pos, pacman_pos, collision, pellets_done, valid)
            rewards.append(r)
        if collision and self.shared_collision:
            for i in range(len(rewards)):
                rewards[i] = max(rewards[i], REWARD_COLLISION)
        return rewards

    def reset_all(self):
        for h in self.reward_helpers:
            h.reset_episode()
        for agent in self.agents:
            agent.reset_episode()

    def get_epsilon(self):
        return float(np.mean([a.epsilon for a in self.agents])) if self.agents else 0.0

    def uses_epsilon(self):
        return True

    def save_all(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save({"q_network": self.shared_dqn.q_network.state_dict(), "target_network": self.shared_dqn.target_network.state_dict()}, os.path.join(directory, "shared_dqn.pt"))

    def load_all(self, directory):
        path = os.path.join(directory, "shared_dqn.pt")
        if os.path.exists(path):
            payload = torch.load(path)
            if isinstance(payload, dict) and "q_network" in payload:
                self.shared_dqn.q_network.load_state_dict(payload["q_network"])
                self.shared_dqn.target_network.load_state_dict(payload.get("target_network", payload["q_network"]))
            else:
                self.shared_dqn.q_network.load_state_dict(payload)
                self.shared_dqn.target_network.load_state_dict(self.shared_dqn.q_network.state_dict())
