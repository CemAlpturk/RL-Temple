from typing import Callable

import torch.nn as nn


def get_activation(activation: str | None) -> Callable:
    if activation is None or activation == "identity":
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f"Unknown activation function: {activation}")
