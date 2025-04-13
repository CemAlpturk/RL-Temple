from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from rl_temple.agents.base_agent import BaseAgent
from rl_temple.utils.replay_buffer import ReplayBuffer
from rl_temple.models.factory import make_model


class DQNAgent(BaseAgent):

    def __init__(
        self,
        action_dim: int,
        model_config: dict[str, Any],
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        device=None,
    ) -> None:

        self.device: torch.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_net = make_model(model_config).to(self.device)
        self.target_net = make_model(model_config).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps: int = 0

    def select_action(self, state, explore: bool = True, return_info: bool = False):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        with torch.no_grad():
            q_values: torch.Tensor = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done, **kwargs):
        self.buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute Q(s, a)
        q_values: torch.Tensor = self.q_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values: torch.Tensor = self.target_net(next_states)
            max_next_q = next_q_values.max(1)[0]
            target = rewards + (1 - dones) * self.gamma * max_next_q

        # Loss and update
        loss: torch.Tensor = nn.MSELoss()(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodic target network update
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def end_episode(self):
        # Not needed
        pass
