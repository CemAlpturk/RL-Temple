from abc import ABC, abstractmethod

import gymnasium as gym

from rl_temple.agents.base_agent import BaseAgent


class AgentRunner(ABC):

    def __init__(self, agent: BaseAgent, env: gym.Env) -> None:
        self.agent = agent
        self.env = env

    @abstractmethod
    def run_episode(self, max_steps: int, explore: bool = True) -> dict[str, float]:
        """
        Run one episode and return episode stats.
        """
        pass
