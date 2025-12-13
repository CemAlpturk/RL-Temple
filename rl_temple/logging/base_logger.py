from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    """Abstract base class for loggers to ensure consistent interface."""

    @abstractmethod
    def log_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: Any | None = None,
    ) -> None:
        """Log a scalar value."""
        pass

    @abstractmethod
    def log_video(
        self,
        tag: str,
        vid_tensor: Any,
        global_step: Any,
        fps: int = 4,
    ) -> None:
        """Log a video tensor."""
        pass

    @abstractmethod
    def save_model(
        self,
        checkpoint: dict[str, Any],
        global_step: int,
    ) -> None:
        """Save a model checkpoint."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger and cleanup resources."""
        pass
