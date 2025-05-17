from typing import Any
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def select_action(
        self, state: Any, explore: bool = True, return_info: bool = False
    ) -> Any:
        """
        Selects an action given a state.
        """
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done, **kwargs) -> None:
        """
        Store transition in replay memory or buffer.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        Perform a learning update.
        """
        pass

    def end_episode(self) -> dict[str, float]:
        """
        Optional hook: called at the end of an episode.
        """
        return {}
