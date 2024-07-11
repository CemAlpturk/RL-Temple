from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import get_activation


class CNN(nn.module):

    def __init__(
        self,
        input_channels: int,
        conv_layers_config: List[Tuple[int, int, Optional[int], Optional[int]]],
        fc_layers_config: List[int],
        num_classes: int,
        activation: str | None = "relu",
    ) -> None:
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        in_channels = input_channels

        # Convolutional layers
        for config in conv_layers_config:
            out_channels, kernel_size, pool_kernel_size, pool_stride = config
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            in_channels = out_channels

            if pool_kernel_size is not None and pool_stride is not None:
                self.pool_layers.append(
                    nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
                )
            else:
                self.pool_layers.append(None)

        self.fc_layers = nn.ModuleList()
        in_features = self._calculate_conv_output(conv_layers_config)

        # Fully connected layers
        for out_features in fc_layers_config:
            self.fc_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        self.output_layer = nn.Linear(in_features, num_classes)

        self.activation = get_activation(activation)

    def _calculate_conv_output(self, input_size: Tuple[int, int]) -> int:
        x = torch.rand(1, *input_size)
        for layer in self.conv_layers:
            x = layer(x)
        return x.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
