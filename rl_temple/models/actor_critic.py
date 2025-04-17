from typing import Sequence

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
from torch import distributions as dist

from .mlp import MLP


class Actor(nn.Module):

    def _distribution(self, obs: torch.Tensor) -> dist.Distribution:
        raise NotImplementedError

    def _log_prob_from_distribution(
        self,
        pi: torch.distributions.Distribution,
        act: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor | None = None,
    ) -> tuple[dist.Distribution, torch.Tensor | None]:
        pi = self._distribution(obs)
        logp_a = None

        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.logits_net = MLP(
            input_size=obs_dim,
            output_size=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def _distribution(self, obs: torch.Tensor) -> dist.Categorical:
        logits: torch.Tensor = self.logits_net(obs)
        return dist.Categorical(logits=logits)

    def _log_prob_from_distribution(
        self,
        pi: torch.distributions.Distribution,
        act: torch.Tensor,
    ) -> torch.Tensor:
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = MLP(
            input_size=obs_dim,
            output_size=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def _distribution(self, obs: torch.Tensor) -> dist.Normal:
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return dist.Normal(mu, std)

    def _log_prob_from_distribution(
        self,
        pi: dist.Distribution,
        act: torch.Tensor,
    ) -> torch.Tensor:
        return pi.log_prob(act).sum(dim=-1)


class MLPCritic(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.v_net = MLP(
            input_size=obs_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ) -> None:

        super().__init__()

        assert observation_space.shape is not None
        obs_dim = observation_space.shape[0]

        # Policy builder
        if isinstance(action_space, gym.spaces.Discrete):
            act_dim = int(action_space.n)
            self.pi = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activation)

        elif isinstance(action_space, gym.spaces.Box):
            act_dim = int(action_space.shape[0])
            self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)

        else:
            raise NotImplementedError

        # Value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v: torch.Tensor = self.v(obs)
        return a, v, logp_a

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.step(obs)[0]
