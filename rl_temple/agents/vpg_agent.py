from typing import Callable

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from rl_temple.agents.base_agent import BaseAgent
from rl_temple.utils.rollout_buffer import RolloutBuffer
from rl_temple.models.actor_critic import ActorCritic


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return returns, advantages


def _compute_rewards_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rewards_to_go = np.zeros_like(rewards)
    running_sum = 0.0
    for i in reversed(range(len(rewards))):
        running_sum = rewards[i] + gamma * running_sum
        rewards_to_go[i] = running_sum
    return rewards_to_go


class VPGAgent(BaseAgent):

    def __init__(
        self,
        model: ActorCritic,
        gamma: float = 0.99,
        lam: float = 0.95,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        vf_lr_decay: float = 0.5,
        vf_lr_patience: int = 10,
        vf_lr_threshold: float = 1e-4,
        vf_early_stopping_patience: int = 10,
        vf_early_stopping_tol: float = -float("inf"),
        batch_size: int = 4,
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

        self.batch_size = batch_size
        self.ep_count = 0
        self.train_v_iters = train_v_iters

        self.pi_optimizer = optim.Adam(self.model.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(self.model.v.parameters(), lr=vf_lr)

        self.vf_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.vf_optimizer,
            mode="min",
            factor=vf_lr_decay,
            patience=vf_lr_patience,
            threshold=vf_lr_threshold,
        )
        self.vf_early_stopping_patience = vf_early_stopping_patience
        self.vf_early_stopping_tol = vf_early_stopping_tol

        postprocess_fn = self._get_postprocess_fn()
        self.buffer = RolloutBuffer(postprocess_fn=postprocess_fn)

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

        states = torch.tensor(
            np.concatenate([ep["states"] for ep in episodes]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            np.concatenate([ep["actions"] for ep in episodes]),
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.tensor(
            np.concatenate([ep["advantages"] for ep in episodes]),
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.tensor(
            np.concatenate([ep["returns"] for ep in episodes]),
            dtype=torch.float32,
            device=self.device,
        )

        # Update policy
        policy_loss = self._train_pi(states, actions, advantages)

        # Update value function
        returns = returns.detach()
        value_loss, train_v_steps = self._train_v(states, returns)
        self.vf_scheduler.step(value_loss)
        vf_lr = self.vf_optimizer.param_groups[0]["lr"]

        stats = {
            "loss_pi": policy_loss,
            "loss_v": value_loss,
            "lr_v": vf_lr,
            "v_train_steps": train_v_steps,
        }
        return stats

    def _train_pi(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> float:
        self.pi_optimizer.zero_grad()

        _, log_probs = self.model.pi(states, actions)
        policy_loss: torch.Tensor = -(log_probs * advantages).mean()

        policy_loss.backward()
        self.pi_optimizer.step()

        return policy_loss.item()

    def _train_v(self, states: torch.Tensor, returns: torch.Tensor) -> tuple[float, int]:
        best_loss = float("inf")
        steps_without_improvement = 0

        for step in range(self.train_v_iters):
            value_preds = self.model.v(states).squeeze()
            value_loss = F.mse_loss(value_preds, returns)

            self.vf_optimizer.zero_grad()
            value_loss.backward()
            self.vf_optimizer.step()

            if best_loss - value_loss.item() > self.vf_early_stopping_tol:
                best_loss = value_loss.item()
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if steps_without_improvement >= self.vf_early_stopping_patience:
                break
        return value_loss.item(), step + 1

    def _get_postprocess_fn(
        self,
    ) -> Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]:

        gamma = self.gamma
        lam = self.lam

        def postprocess_fn(episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            rewards = episode["rewards"]
            values = episode["values"]
            dones = episode["dones"]

            # Compute GAE
            returns, advantages = _compute_gae(rewards, values, dones, gamma, lam)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update the episode dict
            episode["returns"] = returns
            episode["advantages"] = advantages

            return episode

        return postprocess_fn
