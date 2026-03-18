"""
QMIX (Factorized Multi-Agent Q-Learning) for Ghost Control.
"""

import os
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rl_utils import QLearningGhost, ReplayBuffer, REWARD_COLLISION, REPEAT_PATTERN_WINDOW, REPEAT_PATTERN_THRESHOLD, REWARD_REPEAT


class HyperNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.fc1_w = nn.Linear(state_dim, n_agents * hidden_dim)
        self.fc1_b = nn.Linear(state_dim, hidden_dim)
        self.fc2_w = nn.Linear(state_dim, hidden_dim)
        self.fc2_b = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state):
        fc1_w = torch.abs(self.fc1_w(state)).view(-1, self.hidden_dim, self.n_agents)
        fc1_b = self.fc1_b(state)
        fc2_w = torch.abs(self.fc2_w(state)).view(-1, 1, self.hidden_dim)
        fc2_b = self.fc2_b(state)
        return fc1_w, fc1_b, fc2_w, fc2_b


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64):
        super().__init__()
        self.hypernet = HyperNetwork(state_dim, n_agents, hidden_dim)

    def forward(self, individual_q, global_state):
        fc1_w, fc1_b, fc2_w, fc2_b = self.hypernet(global_state)
        hidden = torch.relu(torch.bmm(fc1_w, individual_q.unsqueeze(-1)).squeeze(-1) + fc1_b)
        return torch.bmm(fc2_w, hidden.unsqueeze(-1)).squeeze(-1) + fc2_b


class IndividualQNetwork(nn.Module):
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


class QMIXAgent:
    def __init__(self, ghost_id, state_dim, action_dim, hidden_sizes, lr, gamma, tau):
        self.ghost_id = ghost_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.q_network = IndividualQNetwork(state_dim, action_dim, hidden_sizes)
        self.target_network = IndividualQNetwork(state_dim, action_dim, hidden_sizes)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.total_reward = 0
        self.episode_rewards = []

    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_dim))
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def soft_update_target(self):
        for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def decay_epsilon(self):
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        self.total_reward = 0


class MultiGhostQMIX:
    def __init__(self, num_ghosts, state_dim, action_dim, config, repeat_window=REPEAT_PATTERN_WINDOW, repeat_threshold=REPEAT_PATTERN_THRESHOLD, repeat_penalty=REWARD_REPEAT):
        cfg = config.get("qmix", {})
        hidden = cfg.get("hidden_sizes", [128, 128])
        mixing_hidden = cfg.get("mixing_hidden", 64)
        self.num_ghosts = num_ghosts
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.get("gamma", 0.95)
        self.tau = cfg.get("tau", 0.005)
        self.batch_size = cfg.get("batch_size", 128)
        self.buffer_size = cfg.get("buffer_size", 100000)
        self.update_every = cfg.get("update_every", 4)
        self.lr = cfg.get("lr", 0.0005)
        self.agents = [QMIXAgent(i, state_dim, action_dim, hidden, self.lr, self.gamma, self.tau) for i in range(num_ghosts)]
        global_state_dim = state_dim * num_ghosts
        self.mixing_network = MixingNetwork(num_ghosts, global_state_dim, mixing_hidden)
        self.target_mixing_network = MixingNetwork(num_ghosts, global_state_dim, mixing_hidden)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(self.buffer_size)
        self.timestep = 0
        self.shared_collision = True
        self.reward_helpers = [QLearningGhost(ghost_id=i, repeat_window=repeat_window, repeat_threshold=repeat_threshold, repeat_penalty=repeat_penalty) for i in range(num_ghosts)]
        self.epsilon_start = cfg.get("epsilon_start", 1.0)
        self.epsilon_min = cfg.get("epsilon_min", 0.05)
        self.epsilon_decay = cfg.get("epsilon_decay", 0.995)
        for agent in self.agents:
            agent.epsilon = self.epsilon_start
            agent.epsilon_min = self.epsilon_min
            agent.epsilon_decay = self.epsilon_decay

    def _build_global_state(self, states):
        return np.concatenate([np.array(s, dtype=np.float32) for s in states])

    def get_actions(self, states, training=True):
        return [agent.select_action(np.array(state, dtype=np.float32), training=training) for agent, state in zip(self.agents, states)]

    def update_all(self, states, actions, rewards, next_states, done):
        global_state = self._build_global_state(states)
        global_next_state = self._build_global_state(next_states)
        transition = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "global_state": global_state, "global_next_state": global_next_state, "done": done}
        self.buffer.push(transition, actions[0], sum(rewards), global_next_state, done)
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
        batch = self._sample_batch()
        if batch is None:
            return
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = batch
        total_loss = 0
        for i in range(len(states_batch)):
            states, actions, rewards, next_states, done = states_batch[i], actions_batch[i], rewards_batch[i], next_states_batch[i], dones_batch[i]
            global_state = torch.FloatTensor(self._build_global_state(states)).unsqueeze(0)
            global_next_state = torch.FloatTensor(self._build_global_state(next_states)).unsqueeze(0)
            individual_q = torch.stack([self.agents[j].q_network(torch.FloatTensor(np.array(states[j], dtype=np.float32)).unsqueeze(0))[0, actions[j]] for j in range(self.num_ghosts)]).unsqueeze(0)
            current_q = self.mixing_network(individual_q, global_state)
            with torch.no_grad():
                next_individual_q = torch.stack([torch.max(self.agents[j].target_network(torch.FloatTensor(np.array(next_states[j], dtype=np.float32)).unsqueeze(0)), dim=1)[0].squeeze() for j in range(self.num_ghosts)]).unsqueeze(0)
                next_q = self.target_mixing_network(next_individual_q, global_next_state)
                target_q = torch.FloatTensor([sum(rewards)]).unsqueeze(1) + self.gamma * next_q * (1 - torch.FloatTensor([done]).unsqueeze(1))
            total_loss += nn.MSELoss()(current_q, target_q)
        avg_loss = total_loss / len(states_batch)
        for agent in self.agents:
            agent.optimizer.zero_grad()
        self.mixing_optimizer.zero_grad()
        avg_loss.backward()
        for agent in self.agents:
            agent.optimizer.step()
        self.mixing_optimizer.step()
        for agent in self.agents:
            agent.soft_update_target()
        for tp, p in zip(self.target_mixing_network.parameters(), self.mixing_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _sample_batch(self):
        if len(self.buffer) < self.batch_size:
            return None
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer.buffer[i] for i in idx]
        return [b[0]["states"] for b in batch], [b[0]["actions"] for b in batch], [b[0]["rewards"] for b in batch], [b[0]["next_states"] for b in batch], [b[0]["done"] for b in batch]

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
        for i, agent in enumerate(self.agents):
            torch.save(agent.q_network.state_dict(), os.path.join(directory, f"ghost_{i}_qmix_q.pt"))
        torch.save({"mixing": self.mixing_network.state_dict(), "target_mixing": self.target_mixing_network.state_dict()}, os.path.join(directory, "qmix_mixing.pt"))

    def load_all(self, directory):
        for i, agent in enumerate(self.agents):
            path = os.path.join(directory, f"ghost_{i}_qmix_q.pt")
            if os.path.exists(path):
                agent.q_network.load_state_dict(torch.load(path))
                agent.target_network.load_state_dict(torch.load(path))
        mixing_path = os.path.join(directory, "qmix_mixing.pt")
        if os.path.exists(mixing_path):
            payload = torch.load(mixing_path)
            if isinstance(payload, dict):
                self.mixing_network.load_state_dict(payload.get("mixing", payload))
                self.target_mixing_network.load_state_dict(payload.get("target_mixing", payload.get("mixing", payload)))
            else:
                self.mixing_network.load_state_dict(payload)
                self.target_mixing_network.load_state_dict(payload)
