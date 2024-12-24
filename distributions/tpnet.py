from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


@dataclass
class TensorParamNetConfig:
    """Configuration for tensor parameter network.

    Attributes:
        in_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension for tensor parameters
        num_layers: Number of layers in the network
        activation: Activation function between layers ("relu", "gelu", "tanh", or None)
        positivity_func: Function ensuring parameter positivity ("sq", "abs", "exp")
        dropout: Dropout probability between layers
        use_layer_norm: Whether to use layer normalization
    """

    in_dim: int = 768
    hidden_dim: int = 512
    out_dim: int = 1
    num_layers: int = 2
    activation: Optional[str] = "relu"
    positivity_func: Literal["sq", "abs", "exp"] = "exp"
    dropout: float = 0.0
    use_layer_norm: bool = False


class TensorParamNet(nn.Module):
    def __init__(self, config: TensorParamNetConfig):
        super().__init__()

        self.activation_func = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            None: nn.Identity(),
        }[config.activation]

        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[config.positivity_func]

        layers = []
        current_dim = config.in_dim

        for i in range(config.num_layers):
            out_features = (
                config.out_dim if i == config.num_layers - 1 else config.hidden_dim
            )

            if config.use_layer_norm:
                layers.append(nn.LayerNorm(current_dim))

            layers.append(nn.Linear(current_dim, out_features))

            if i < config.num_layers - 1:
                if config.activation is not None:
                    layers.append(self.activation_func)
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))

            current_dim = out_features

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.network(x)
        return self.positivity_func(params)
