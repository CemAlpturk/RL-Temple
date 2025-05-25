import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from rl_temple.agents.base_agent import BaseAgent
from rl_temple.utils.rollout_buffer import RolloutBuffer
from rl_temple.models.actor_critic import ActorCritic


class VPGAgent(BaseAgent):

    def __init__(
        self,
        model: ActorCritic,
        gamma: float = 0.99,
        lam: float = 0.95,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        batch_size: int = 4,
        device=None,
    ) -> None:

        self.device: torch.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.lam = lam

        self.model = model
        self.model.to(self.device)

        self.batch_size = batch_size
        self.ep_count = 0

        self.pi_optimizer = optim.Adam(self.model.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(self.model.v.parameters(), lr=vf_lr)

        self.buffer = RolloutBuffer()

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        return_info: bool = False,
    ):

        # Add batch dim
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        act, value, logprob = self.model.step(state_t)

        if return_info:
            return act.cpu().numpy()[0], (logprob.item(), value.item())

        return act.cpu().numpy()[0]

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.buffer.add((state, action, reward, log_prob, value, done))

    def update(self) -> None:
        # No update
        pass

    def end_episode(self) -> dict[str, float]:
        self.buffer.end_episode()
        if len(self.buffer) == self.batch_size:
            stats = self._learn()
            self.buffer.clear()
            return stats

        return {}

    def _learn(self) -> dict[str, float]:
        episodes = self.buffer.get()

        all_states, all_actions = [], []
        all_returns, all_advantages = [], []

        for episode in episodes:
            states = episode["states"]
            actions = episode["actions"]
            rewards = episode["rewards"]
            values = episode["values"]
            dones = episode["dones"]
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            values = torch.tensor(values, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            # Use GAE for returns and advantages
            returns, advantages = self._compute_gae(rewards, values, dones)

            # Stash
            all_states.append(states)
            all_actions.append(actions)
            all_returns.append(returns)
            all_advantages.append(advantages)

        states = torch.cat(all_states, dim=0)  # (N, *state_dim)
        actions = torch.cat(all_actions, dim=0)  # (N, *action_dim)
        returns = torch.cat(all_returns, dim=0)  # (N,)
        advantages = torch.cat(all_advantages, dim=0)  # (N,)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy
        _, log_probs = self.model.pi(states, actions)
        policy_loss: torch.Tensor = -(log_probs * advantages).mean()

        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        self.pi_optimizer.step()

        # Update value function
        returns = returns.detach()
        value_preds = self.model.v(states).squeeze()  # (N,)
        value_loss = F.mse_loss(value_preds, returns)

        self.vf_optimizer.zero_grad()
        value_loss.backward()
        self.vf_optimizer.step()

        stats = {
            "loss_pi": policy_loss.item(),
            "loss_v": value_loss.item(),
        }
        return stats

    def _compute_rewards_to_go(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards_to_go = torch.zeros_like(rewards)
        running_sum = 0.0
        for i in reversed(range(len(rewards))):
            running_sum = rewards[i] + self.gamma * running_sum
            rewards_to_go[i] = running_sum
        return rewards_to_go

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return returns, advantages
