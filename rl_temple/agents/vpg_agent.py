import torch
import torch.optim as optim

import numpy as np

from rl_temple.agents.base_agent import BaseAgent
from rl_temple.utils.rollout_buffer import RolloutBuffer
from rl_temple.models.actor_critic import MLPActorCritic


class VPGAgent(BaseAgent):

    def __init__(
        self,
        model: MLPActorCritic,
        gamma: float = 0.99,
        lam: float = 0.95,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_v_iters: int = 80,
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
            return act.item(), (logprob.item(), value.item())

        return act.item()

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

    def end_episode(self) -> None:
        self._learn()
        self.buffer.clear()

    def _learn(self) -> None:
        states, actions, rewards, log_probs_old, values, dones = self.buffer.get()

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        log_probs_old = torch.tensor(
            log_probs_old, dtype=torch.float32, device=self.device
        )
        values = torch.tensor(values, dtype=torch.float32, device=self.device)

        # Rewards to go
        rewards_to_go = self._compute_rewards_to_go(rewards)
        rewards = torch.tensor(rewards_to_go, dtype=torch.float32, device=self.device)

        # Policy loss
        self.pi_optimizer.zero_grad()
        pi, logp = self.model.pi(states, actions)
        loss_pi = -(logp * rewards).mean()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Value loss
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = ((self.model.v(states) - rewards) ** 2).mean()
            loss_v.backward()
            self.vf_optimizer.step()

    def _compute_rewards_to_go(self, rewards: list[float]) -> list[float]:
        rewards_to_go = [0.0] * len(rewards)
        running_sum = 0.0
        for i in reversed(range(len(rewards))):
            running_sum = rewards[i] + self.gamma * running_sum
            rewards_to_go[i] = running_sum
        return rewards_to_go
