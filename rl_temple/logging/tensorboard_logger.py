import os
import datetime
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:

    def __init__(self, log_dir: str | None = None) -> None:

        if log_dir is None:
            log_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../runs")
            )

        self.log_dir = log_dir

        self._init_directory()

        self.writer = SummaryWriter(self.tb_dir)

    def _init_directory(self) -> None:
        # Create base log dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Create run dir
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(self.log_dir, timestamp)
        os.mkdir(run_dir)
        self.run_dir = run_dir

        # Create directory for tensorboard logs
        self.tb_dir = os.path.join(run_dir, "tensorboard")
        os.mkdir(self.tb_dir)

        # Create directory for model checkpoints
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.mkdir(self.ckpt_dir)

    def log_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: Any | None = None,
    ) -> None:
        self.writer.add_scalar(tag, scalar_value, global_step)

    def log_video(
        self,
        tag: str,
        vid_tensor: Any,
        global_step: Any,
        fps: int = 4,
    ) -> None:
        self.writer.add_video(tag, vid_tensor, global_step, fps=fps)

    def save_model(
        self,
        checkpoint: dict[str, Any],
        global_step: int,
    ) -> None:
        torch.save(checkpoint, os.path.join(self.ckpt_dir, f"model_{global_step}.pt"))

    def close(self):
        self.writer.close()
