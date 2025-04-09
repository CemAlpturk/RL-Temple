from typing import Any
import random

import numpy as np


class ReplayBuffer:

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[tuple] = []
        self.pos: int = 0

    def add(
        self,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        done: Any,
    ) -> None:
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
