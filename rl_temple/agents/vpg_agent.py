import torch
import torch.optim as optim

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
        train_v_iters: int = 80,
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

        self.train_v_iters = train_v_iters
        self.batch_size = batch_size
        self.ep_count = 0

        self.pi_optimizer = optim.Adam(self.model.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(self.model.v.parameters(), lr=vf_lr)

        self.pi_optimizer.zero_grad()
        self.vf_optimizer.zero_grad()

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
        state,
        action,
        reward,
        next_state,
        done,
        log_prob=None,
        value=None,
    ) -> None:
        self.buffer.add((state, action, reward, log_prob, value, done))

    def update(self) -> None:
        # No update
        pass

    def end_episode(self) -> dict[str, float]:
        stats = self._learn()
        self.buffer.clear()
        return stats

    def _learn(self) -> dict[str, float]:
        states, actions, rewards, log_probs_old, values, dones = self.buffer.get()
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), device=self.device)
        log_probs_old = torch.tensor(
            log_probs_old, dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        returns = self._compute_rewards_to_go(rewards)
        advantages = returns - values

        # Policy loss
        _, logp = self.model.pi(states, actions)
        loss_pi = -(logp * advantages).mean()
        loss_pi /= self.batch_size
        loss_pi.backward()

        self.ep_count += 1
        if self.ep_count % self.batch_size == 0:
            self.pi_optimizer.step()
            self.ep_count = 0
            self.pi_optimizer.zero_grad()

        # Value loss
        returns = returns.detach()
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = ((self.model.v(states) - returns) ** 2).mean()
            loss_v.backward()
            self.vf_optimizer.step()

        stats = {
            "loss_pi": loss_pi.item(),
            "loss_v": loss_v.item(),
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
        """
        Computes GAE-Lambda advantage estimates and TD-lambda returns.
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        last_adv = 0.0
        vls = torch.cat((values, torch.zeros(1, device=self.device)))
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * vls[t + 1] * mask - vls[t]
            last_adv = delta + self.gamma * self.lam * mask * last_adv
            advantages[t] = last_adv
        returns = advantages + vls[:-1]

        return advantages, returns
