from typing import Sequence
import torch.nn as nn

from .utils import get_activation


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int] = (),
        activation: str | None = "relu",
        final_activation: str | None = None,
        bias: bool = True,
        final_bias: bool = True,
    ) -> None:
        super().__init__()

        features = [input_size] + list(hidden_sizes) + [output_size]
        layers = [
            nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias if i < len(features) - 2 else final_bias,
            )
            for i, (in_features, out_features) in enumerate(
                zip(features[:-1], features[1:])
            )
        ]
        self.layers = nn.ModuleList(layers)
        self.activation = get_activation(activation)
        self.final_activation = get_activation(final_activation)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.final_activation(self.layers[-1](x))
