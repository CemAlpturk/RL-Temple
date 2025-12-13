import os
import datetime
from typing import Any

import torch
import numpy as np

import wandb

from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    """Weights & Biases logger with the same interface as TensorboardLogger."""

    def __init__(
        self,
        project: str = "rl-temple",
        entity: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        log_dir: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        mode: str = "online",
        **kwargs,
    ) -> None:
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            entity: W&B entity (username or team name)
            name: Run name (defaults to timestamp if None)
            config: Hyperparameters and configuration to log
            log_dir: Local directory for saving artifacts (defaults to runs/)
            tags: List of tags for organizing runs
            notes: Longer description of the run
            mode: "online", "offline", or "disabled"
            **kwargs: Additional arguments passed to wandb.init()
        """

        if log_dir is None:
            log_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../runs")
            )

        self.log_dir = log_dir
        self._init_directory()

        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config or {},
            dir=self.run_dir,
            tags=tags,
            notes=notes,
            mode=mode,
            **kwargs,
        )

    def _init_directory(self) -> None:
        """Create directory structure for local artifacts."""
        # Create base log dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Create run dir
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(self.log_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir

        # Create directory for model checkpoints
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def log_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: Any | None = None,
    ) -> None:
        """
        Log a scalar value to W&B.

        Args:
            tag: Metric name (e.g., "train/reward")
            scalar_value: Scalar value to log
            global_step: Training step or episode number
        """
        log_dict = {tag: scalar_value}
        if global_step is not None:
            log_dict["step"] = global_step
        wandb.log(log_dict, step=global_step)

    def log_video(
        self,
        tag: str,
        vid_tensor: Any,
        global_step: Any,
        fps: int = 4,
    ) -> None:
        """
        Log a video to W&B.

        Args:
            tag: Video name
            vid_tensor: Video tensor of shape (N, T, C, H, W) as torch.uint8
            global_step: Training step or episode number
            fps: Frames per second
        """
        # Convert tensor to numpy if needed
        if isinstance(vid_tensor, torch.Tensor):
            vid_array = vid_tensor.cpu().numpy()
        else:
            vid_array = np.array(vid_tensor)

        # W&B expects (T, H, W, C) format
        # Input is (N, T, C, H, W), take first batch
        if len(vid_array.shape) == 5:
            vid_array = vid_array[0]  # (T, C, H, W)

        # Transpose to (T, H, W, C)
        vid_array = vid_array.transpose(0, 2, 3, 1)

        wandb.log(
            {tag: wandb.Video(vid_array, fps=fps, format="mp4")},
            step=global_step,
        )

    def save_model(
        self,
        checkpoint: dict[str, Any],
        global_step: int,
    ) -> None:
        """
        Save model checkpoint locally and as W&B artifact.

        Args:
            checkpoint: Model state dict and other info
            global_step: Training step or episode number
        """
        # Save locally
        checkpoint_path = os.path.join(self.ckpt_dir, f"model_{global_step}.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save as W&B artifact
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            description=f"Model checkpoint at step {global_step}",
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    def log_config(self, config: dict[str, Any]) -> None:
        """
        Log hyperparameters and configuration.

        Args:
            config: Dictionary of hyperparameters
        """
        wandb.config.update(config)

    def log_histogram(
        self,
        tag: str,
        values: Any,
        global_step: Any | None = None,
    ) -> None:
        """
        Log a histogram of values (useful for weights/gradients).

        Args:
            tag: Histogram name
            values: Array-like or tensor of values
            global_step: Training step or episode number
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        wandb.log(
            {tag: wandb.Histogram(values)},
            step=global_step,
        )

    def log_table(
        self,
        tag: str,
        data: list[list[Any]],
        columns: list[str],
        global_step: Any | None = None,
    ) -> None:
        """
        Log structured data as a table.

        Args:
            tag: Table name
            data: List of rows
            columns: Column names
            global_step: Training step or episode number
        """
        table = wandb.Table(data=data, columns=columns)
        wandb.log({tag: table}, step=global_step)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        global_step: Any | None = None,
    ) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value
            global_step: Training step or episode number
        """
        wandb.log(metrics, step=global_step)

    def watch_model(
        self,
        model: torch.nn.Module,
        log_freq: int = 100,
        log: str = "all",
    ) -> None:
        """
        Watch a model to log gradients and parameters.

        Args:
            model: PyTorch model to watch
            log_freq: How often to log (in training steps)
            log: What to log - "gradients", "parameters", "all", or None
        """
        wandb.watch(model, log=log, log_freq=log_freq)

    def close(self) -> None:
        """Finish the W&B run."""
        if self.run is not None:
            self.run.finish()
