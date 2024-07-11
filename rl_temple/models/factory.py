from typing import Any
import torch.nn as nn
from .mlp import MLP


def make_model(model_config: dict[str, Any]) -> nn.Module:
    if model_config["type"] == "mlp":
        return MLP(**model_config["args"])
    else:
        raise NotImplementedError(f"Unknown model type: {model_config['type']}")
