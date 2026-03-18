"""
PPO (Proximal Policy Optimization) for Ghost Control.
"""

import os
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from rl_utils import QLearningGhost, REWARD_COLLISION, REPEAT_PATTERN_WINDOW, REPEAT_PATTERN_THRESHOLD, REWARD_REPEAT


class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_sizes):
        super().__init__()
        layers = []
        last = input_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(last, size), nn.ReLU()])
            last = size
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, action_dim)
        self.value_head = nn.Linear(last, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


class MultiGhostPPO:
    def __init__(self, num_ghosts, state_dim, action_dim, config, repeat_window=REPEAT_PATTERN_WINDOW, repeat_threshold=REPEAT_PATTERN_THRESHOLD, repeat_penalty=REWARD_REPEAT):
        cfg = config.get("ppo", {})
        hidden = cfg.get("hidden_sizes", [128, 128])
        self.num_ghosts = num_ghosts
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.get("gamma", 0.99)
        self.gae_lambda = cfg.get("gae_lambda", 0.95)
        self.clip_epsilon = cfg.get("clip_epsilon", 0.2)
        self.value_coef = cfg.get("value_coef", 0.5)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.update_epochs = cfg.get("update_epochs", 4)
        self.minibatch_size = cfg.get("minibatch_size", 256)
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)
        self.network = PPOActorCritic(state_dim, action_dim, hidden)
        self.optimizer = optim.Adam(self.network.parameters(), lr=cfg.get("lr", 0.0003))
        self.shared_collision = True
        self.trajectories = [[] for _ in range(num_ghosts)]
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0
        self.total_updates = 0
        self.reward_helpers = [QLearningGhost(ghost_id=i, repeat_window=repeat_window, repeat_threshold=repeat_threshold, repeat_penalty=repeat_penalty) for i in range(num_ghosts)]

    def _augment_state(self, state, ghost_id):
        one_hot = np.zeros(self.num_ghosts, dtype=np.float32)
        if 0 <= ghost_id < self.num_ghosts:
            one_hot[ghost_id] = 1.0
        return np.concatenate([np.array(state, dtype=np.float32), one_hot])

    def _policy_output(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(state_t)
        return logits.squeeze(0), float(value.squeeze(0).item())

    def get_actions(self, states, training=True):
        actions = []
        for ghost_id, state in enumerate(states):
            logits, _ = self._policy_output(self._augment_state(state, ghost_id))
            if training:
                action = int(Categorical(logits=logits).sample().item())
            else:
                action = int(torch.argmax(logits).item())
            actions.append(action)
        return actions

    def update_all(self, states, actions, rewards, next_states, done):
        for i in range(min(len(states), self.num_ghosts)):
            augmented = self._augment_state(states[i], i)
            logits, value = self._policy_output(augmented)
            dist = Categorical(logits=logits)
            self.trajectories[i].append({"state": augmented, "action": int(actions[i]), "reward": float(rewards[i]), "next_state": self._augment_state(next_states[i], i), "done": bool(done), "value": value, "old_log_prob": float(dist.log_prob(torch.tensor(actions[i], dtype=torch.long)).item())})
        if done:
            self.finish_episode()

    def finish_episode(self):
        if not any(self.trajectories):
            return
        batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = [], [], [], [], []
        for trajectory in self.trajectories:
            if not trajectory:
                continue
            rewards = np.array([s["reward"] for s in trajectory], dtype=np.float32)
            dones = np.array([s["done"] for s in trajectory], dtype=np.float32)
            values = np.array([s["value"] for s in trajectory], dtype=np.float32)
            next_states = np.array([s["next_state"] for s in trajectory], dtype=np.float32)
            with torch.no_grad():
                _, next_values_t = self.network(torch.FloatTensor(next_states))
                next_values = next_values_t.cpu().numpy().astype(np.float32)
            next_values = next_values * (1.0 - dones)
            advantages = np.zeros_like(rewards, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(len(trajectory))):
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values
            batch_states.extend([s["state"] for s in trajectory])
            batch_actions.extend([s["action"] for s in trajectory])
            batch_old_log_probs.extend([s["old_log_prob"] for s in trajectory])
            batch_returns.extend(returns.tolist())
            batch_advantages.extend(advantages.tolist())
        if not batch_states:
            self.trajectories = [[] for _ in range(self.num_ghosts)]
            return
        states_t = torch.FloatTensor(np.array(batch_states, dtype=np.float32))
        actions_t = torch.LongTensor(np.array(batch_actions, dtype=np.int64))
        old_log_probs_t = torch.FloatTensor(np.array(batch_old_log_probs, dtype=np.float32))
        returns_t = torch.FloatTensor(np.array(batch_returns, dtype=np.float32))
        advantages_t = torch.FloatTensor(np.array(batch_advantages, dtype=np.float32))
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)
        dataset_size = states_t.size(0)
        minibatch_size = max(1, min(self.minibatch_size, dataset_size))
        policy_losses, value_losses, entropies = [], [], []
        for _ in range(self.update_epochs):
            permutation = torch.randperm(dataset_size)
            for start in range(0, dataset_size, minibatch_size):
                indices = permutation[start:start + minibatch_size]
                new_log_probs, entropy, values = self.network.evaluate_actions(states_t[indices], actions_t[indices])
                ratio = torch.exp(new_log_probs - old_log_probs_t[indices])
                unclipped = ratio * advantages_t[indices]
                clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t[indices]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.MSELoss()(values, returns_t[indices])
                entropy_bonus = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_bonus.item()))
        self.last_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        self.last_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
        self.last_entropy = float(np.mean(entropies)) if entropies else 0.0
        self.total_updates += 1
        self.trajectories = [[] for _ in range(self.num_ghosts)]

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
        self.trajectories = [[] for _ in range(self.num_ghosts)]
        for h in self.reward_helpers:
            h.reset_episode()

    def get_epsilon(self):
        return 0.0

    def uses_epsilon(self):
        return False

    def get_training_stats(self):
        return {"policy_loss": self.last_policy_loss, "value_loss": self.last_value_loss, "entropy": self.last_entropy, "total_updates": self.total_updates}

    def save_all(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save({"network": self.network.state_dict(), "optimizer": self.optimizer.state_dict(), "total_updates": self.total_updates, "last_policy_loss": self.last_policy_loss, "last_value_loss": self.last_value_loss, "last_entropy": self.last_entropy}, os.path.join(directory, "shared_ppo.pt"))

    def load_all(self, directory):
        path = os.path.join(directory, "shared_ppo.pt")
        if os.path.exists(path):
            payload = torch.load(path, map_location=torch.device("cpu"))
            if isinstance(payload, dict) and "network" in payload:
                self.network.load_state_dict(payload["network"])
                if payload.get("optimizer"):
                    self.optimizer.load_state_dict(payload["optimizer"])
                self.total_updates = payload.get("total_updates", 0)
                self.last_policy_loss = payload.get("last_policy_loss", 0.0)
                self.last_value_loss = payload.get("last_value_loss", 0.0)
                self.last_entropy = payload.get("last_entropy", 0.0)
            else:
                self.network.load_state_dict(payload)
