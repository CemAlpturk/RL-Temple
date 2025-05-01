from typing import Any

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

from rl_temple.agents.base_agent import BaseAgent
from rl_temple.utils.rollout_buffer import RolloutBuffer
from rl_temple.models.factory import make_model


class PPOAgent(BaseAgent):

    def __init__(
        self,
        action_dim: int,
        model_config: dict[str, Any],
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        lr: float = 3e-4,
        epochs: int = 4,
        batch_size: int = 64,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        # Networks
        self.actor = make_model(model_config["actor"]).to(self.device)
        self.critic = make_model(model_config["critic"]).to(self.device)

        combined_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(combined_params, lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(self, state, explore: bool = True, return_info: bool = False):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        logits: torch.Tensor = self.actor(state_t)
        value: torch.Tensor = self.critic(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob: torch.Tensor = dist.log_prob(action)
        if return_info:
            return action.item(), (log_prob.item(), value.item())

        return action.item()

    def remember(
        self, state, action, reward, next_state, done, log_prob=None, value=None
    ):
        self.buffer.add((state, action, reward, log_prob, value, done))

    def update(self):
        # No update
        pass

    def end_episode(self):
        self._learn()
        self.buffer.clear()

    def _learn(self):
        states, actions, rewards, log_probs_old, values, dones = self.buffer.get()

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(
            log_probs_old, dtype=torch.float32, device=self.device
        )
        values = torch.tensor(values, dtype=torch.float32, device=self.device)

        gae = self._compute_gae(rewards, values.cpu().numpy(), dones)
        returns = torch.tensor(gae, dtype=torch.float32, device=self.device)

        advantages = returns - values

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                idx = slice(i, i + self.batch_size)
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]
                batch_old_log_probs = old_log_probs[idx]

                logits = self.actor(batch_states)
                value = self.critic(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * batch_adv, clipped * batch_adv).mean()
                value_loss = F.mse_loss(value.squeeze(-1), batch_returns)
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _compute_gae(self, rewards, values, dones):
        returns, gae = [], 0
        values = list(values) + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])

        return returns
