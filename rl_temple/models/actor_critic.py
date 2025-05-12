from typing import Sequence, Any

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
from torch import distributions as dist

from .mlp import MLP
from .cnn import CNN


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


class CNNCategoricalActor(Actor):

    def __init__(
        self,
        input_channels: int,
        input_shape: tuple[int, int],
        conv_layers: list[dict[str, Any]],
        fc_layers: list[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.logits_net = CNN(
            input_channels=input_channels,
            input_shape=input_shape,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=num_classes,
            activation=activation,
            dropout=dropout,
        )

    def _distribution(self, obs: torch.Tensor) -> dist.Categorical:
        logits: torch.Tensor = self.logits_net(obs)
        return dist.Categorical(logits=logits)

    def _log_prob_from_distribution(
        self, pi: dist.Distribution, act: torch.Tensor
    ) -> torch.Tensor:
        return pi.log_prob(act)


class CNNGaussianActor(Actor):

    def __init__(
        self,
        input_channels: int,
        input_shape: tuple[int, int],
        conv_layers: list[dict[str, Any]],
        fc_layers: list[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        log_std = -0.5 * np.ones(num_classes, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = CNN(
            input_channels=input_channels,
            input_shape=input_shape,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=num_classes,
            activation=activation,
            dropout=dropout,
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


class CNNCritic(nn.Module):

    def __init__(
        self,
        input_channels: int,
        input_shape: tuple[int, int],
        conv_layers: list[dict[str, Any]],
        fc_layers: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.v_net = CNN(
            input_channels=input_channels,
            input_shape=input_shape,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            num_classes=1,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.v_net(obs), -1)


class ActorCritic(nn.Module):
    pi: Actor
    v: nn.Module

    def step(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.step(obs)[0]


class MLPActorCritic(ActorCritic):

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


class CNNActorCritic(ActorCritic):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        conv_layers: list[dict[str, Any]],
        fc_layers: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        assert observation_space.shape is not None
        obs_shape = observation_space.shape

        # Expecting (H, W, C)
        if len(obs_shape) == 2:
            channels = 1
            obs_dim: tuple[int, int] = obs_shape
        elif len(obs_shape) == 3:
            channels = obs_shape[0]
            obs_dim = obs_shape[1:]

        else:
            print(obs_shape)
            raise ValueError("Obs shape must be 2 or 3 dimensional")

        # Policy builder
        if isinstance(action_space, gym.spaces.Discrete):
            act_dim = int(action_space.n)
            self.pi = CNNCategoricalActor(
                input_channels=channels,
                input_shape=obs_dim,
                conv_layers=conv_layers,
                fc_layers=fc_layers,
                num_classes=act_dim,
                activation=activation,
            )

        elif isinstance(action_space, gym.spaces.Box):
            act_dim = int(action_space.shape[0])
            self.pi = CNNGaussianActor(
                input_channels=channels,
                input_shape=obs_dim,
                conv_layers=conv_layers,
                fc_layers=fc_layers,
                num_classes=act_dim,
                activation=activation,
            )

        else:
            raise NotImplementedError

        # Value Function
        self.v = CNNCritic(
            input_channels=channels,
            input_shape=obs_dim,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            activation=activation,
            dropout=dropout,
        )

    @torch.no_grad
    def step(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        pi = self.pi._distribution(obs)
        a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v: torch.Tensor = self.v(obs)
        return a, v, logp_a
