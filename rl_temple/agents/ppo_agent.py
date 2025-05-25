from typing import Callable

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from rl_temple.agents.base_agent import BaseAgent
from rl_temple.utils.rollout_buffer import RolloutBuffer
from rl_temple.models.actor_critic import ActorCritic
from .utils import compute_gae


class PPOAgent(BaseAgent):

    def __init__(
        self,
        model: ActorCritic,
        gamma: float = 0.99,
        lam: float = 0.97,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        n_episodes: int = 1000,
        epochs: int = 10,
        batch_size: int = 64,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        device: torch.device | None = None,
    ) -> None:

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.lam = lam

        self.model = model
        self.model.to(self.device)

        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.ep_count = 0
        self.epochs = epochs

        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

        self.pi_optimizer = optim.Adam(self.model.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(self.model.v.parameters(), lr=vf_lr)

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
        act, value, logp = self.model.step(state_t)

        if return_info:
            return act.cpu().numpy()[0], (logp.item(), value.item())
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
        if len(self.buffer) == self.n_episodes:
            stats = self._learn()
            self.buffer.clear()
            return stats
        return {}

    def _learn(self) -> dict[str, float]:
        episodes = self.buffer.get()

        states = torch.tensor(
            np.concatenate([ep["states"] for ep in episodes]),
            dtype=torch.float32,
        )
        actions = torch.tensor(
            np.concatenate([ep["actions"] for ep in episodes]),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.tensor(
            np.concatenate([ep["log_probs"] for ep in episodes]),
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

        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        continue_training = True
        pi_losses = []
        v_losses = []
        kls = []
        entropies = []
        clip_fracs = []
        for epoch in range(self.epochs):
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for batch in dataloader:
                (
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                ) = (b.to(self.device) for b in batch)

                # Update policy
                self.pi_optimizer.zero_grad()
                loss_pi, pi_info = self._compute_loss_pi(
                    batch_states, batch_actions, batch_old_log_probs, batch_advantages
                )
                kl = pi_info["kl"]
                if kl > 1.5 * self.target_kl:
                    # Early stopping if KL divergence is too high
                    continue_training = False
                    break
                loss_pi.backward()
                self.pi_optimizer.step()

                # Update value function
                self.vf_optimizer.zero_grad()
                loss_v = self._compute_loss_v(batch_states, batch_returns)
                loss_v.backward()
                self.vf_optimizer.step()

                pi_losses.append(loss_pi.item())
                v_losses.append(loss_v.item())
                kls.append(pi_info["kl"])
                entropies.append(pi_info["ent"])
                clip_fracs.append(pi_info["clip_frac"])

            if not continue_training:
                # Early stopping
                break

        stats = {
            "loss_pi": np.mean(pi_losses),
            "loss_v": np.mean(v_losses),
            "KL": np.mean(kls),
            "entropy": np.mean(entropies),
            "clip_frac": np.mean(clip_fracs),
        }

        return stats  # type: ignore

    def _compute_loss_pi(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        # Policy loss
        pi, logp = self.model.pi(states, actions)
        ratio = torch.exp(logp - old_log_probs)
        clip_adv = (
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        )
        loss_pi = -(torch.min(ratio * advantages, clip_adv)).mean()

        # Extra data
        approx_kl = (old_log_probs - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clip_frac = torch.as_tensor(clipped).float().mean().item()
        pi_info = {
            "kl": approx_kl,
            "ent": ent,
            "clip_frac": clip_frac,
        }
        return loss_pi, pi_info

    def _compute_loss_v(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        # Value function loss
        values = self.model.v(states)
        loss_v = F.mse_loss(values.squeeze(-1), returns)
        return loss_v

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
            returns, advantages = compute_gae(rewards, values, dones, gamma, lam)

            # Add to episode
            episode["returns"] = returns
            episode["advantages"] = advantages

            return episode

        return postprocess_fn
