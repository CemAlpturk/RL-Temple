from typing import Union, Any
import torch
import torch.nn as nn

from .utils import get_activation
from .mlp import MLP


class CNN(nn.Module):
    """
    A general-purpose CNN builder.

    Args:
        input_channels (int): Number of channels in the input (e.g., 3 for RGB).
        input_shape (tuple[int, int]): Height and width of extected inputs.
        conv_layers (list[dict[str, Any]]):
            - Dict configs defining convolutional blocks.
        fc_layers (list[int]):
            - Int sizes defining fully-connected blocks.
        num_classes (int): Number of output classes.
        activation (str): Activation class (e.g., nn.ReLU). Default: 'relu'.
        dropout (float): Dropout probability after FC layers (only for int-style FC). Default: 0.0.
    """

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
        self.activation = get_activation(activation)
        self.dropout = dropout

        layers: list[nn.Module] = []
        in_channels = input_channels

        # Conv Layers Setup
        for cfg in conv_layers:
            out_ch: int = cfg["out_channels"]
            k: Union[int, tuple] = cfg.get("kernel_size", 3)
            s: Union[int, tuple] = cfg.get("stride", 1)
            p: Union[int, tuple] = cfg.get("padding", 0)

            layers.append(
                nn.Conv2d(in_channels, out_ch, kernel_size=k, stride=s, padding=p)
            )

            if cfg.get("batch_norm", False):
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(self.activation)
            pool: dict[str, Any] = cfg.get("pooling", {})
            if pool:
                kt: int = pool["kernel_size"]
                st: int = pool.get("stride", kt)
                if pool.get("type", "max") == "max":
                    layers.append(nn.MaxPool2d(kt, stride=st))
                else:
                    layers.append(nn.AvgPool2d(kt, stride=st))

            in_channels = out_ch

        self.conv = nn.Sequential(*layers)

        # Determine flattened size
        h, w = input_shape
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, h, w)
            feat: torch.Tensor = self.conv(dummy)
            flatten_dim = feat.view(1, -1).size(1)

        # FC Setup
        self.fc = MLP(
            input_size=flatten_dim,
            output_size=num_classes,
            hidden_sizes=fc_layers,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.conv(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)
