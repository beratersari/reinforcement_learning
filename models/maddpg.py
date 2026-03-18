"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) for Ghost Control.
"""

import os
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rl_utils import QLearningGhost, ReplayBuffer, REWARD_COLLISION, REPEAT_PATTERN_WINDOW, REPEAT_PATTERN_THRESHOLD, REWARD_REPEAT


class Actor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        last = input_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(last, size), nn.ReLU()])
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
            layers.extend([nn.Linear(last, size), nn.ReLU()])
            last = size
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MADDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_sizes, actor_lr, critic_lr, gamma, tau, epsilon_start, epsilon_min, epsilon_decay):
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

    def select_action(self, state, training=True):
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
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(torch.cat([next_states_t, next_actions], dim=1))
            y = rewards_t + self.gamma * (1 - dones_t) * target_q
        current_q = self.critic(torch.cat([states_t, actions_t], dim=1))
        critic_loss = nn.MSELoss()(current_q, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        actor_loss = -self.critic(torch.cat([states_t, self.actor(states_t)], dim=1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def decay_epsilon(self):
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class MultiGhostMADDPG:
    def __init__(self, num_ghosts, state_dim, action_dim, config, repeat_window=REPEAT_PATTERN_WINDOW, repeat_threshold=REPEAT_PATTERN_THRESHOLD, repeat_penalty=REWARD_REPEAT):
        cfg = config.get("maddpg", {})
        hidden = cfg.get("hidden_sizes", [128, 128])
        self.agents = [MADDPGAgent(state_dim, action_dim, hidden, cfg.get("actor_lr", 0.0005), cfg.get("critic_lr", 0.001), cfg.get("gamma", 0.95), cfg.get("tau", 0.01), cfg.get("epsilon_start", 0.2), cfg.get("epsilon_min", 0.02), cfg.get("epsilon_decay", 0.995)) for _ in range(num_ghosts)]
        self.num_ghosts = num_ghosts
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = ReplayBuffer(cfg.get("buffer_size", 100000))
        self.batch_size = cfg.get("batch_size", 128)
        self.update_every = cfg.get("update_every", 1)
        self.timestep = 0
        self.shared_collision = True
        self.last_action_cont = [np.zeros(action_dim, dtype=np.float32) for _ in range(num_ghosts)]
        self.reward_helpers = [QLearningGhost(ghost_id=i, repeat_window=repeat_window, repeat_threshold=repeat_threshold, repeat_penalty=repeat_penalty) for i in range(num_ghosts)]

    def get_actions(self, states, training=True):
        actions = []
        for i, (agent, state) in enumerate(zip(self.agents, states)):
            action_cont = agent.select_action(np.array(state, dtype=np.float32), training=training)
            self.last_action_cont[i] = action_cont
            actions.append(int(np.argmax(action_cont)))
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
            agent.total_reward = 0
        self.last_action_cont = [np.zeros(self.action_dim, dtype=np.float32) for _ in range(self.num_ghosts)]

    def save_all(self, directory):
        os.makedirs(directory, exist_ok=True)
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(directory, f"ghost_{i}_actor.pt"))
            torch.save(agent.critic.state_dict(), os.path.join(directory, f"ghost_{i}_critic.pt"))

    def load_all(self, directory):
        for i, agent in enumerate(self.agents):
            ap, cp = os.path.join(directory, f"ghost_{i}_actor.pt"), os.path.join(directory, f"ghost_{i}_critic.pt")
            if os.path.exists(ap):
                agent.actor.load_state_dict(torch.load(ap))
            if os.path.exists(cp):
                agent.critic.load_state_dict(torch.load(cp))

    def get_epsilon(self):
        return float(np.mean([a.epsilon for a in self.agents])) if self.agents else 0.0

    def uses_epsilon(self):
        return True
