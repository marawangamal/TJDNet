from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


@dataclass
class TensorParamNetConfig:
    """Configuration for tensor parameter predictor network.

    Attributes:
        in_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension for tensor parameters
        positivity_func: Function ensuring parameter positivity ("sq", "abs", "exp")
    """

    in_dim: int = 768
    hidden_dim: int = 512
    out_dim_encoder: int = 1
    out_dim_decoder: int = 1
    positivity_func: Literal["sq", "abs", "exp"] = "exp"
    use_decoder: bool = True


class TensorParamNet(nn.Module):
    def __init__(self, config: TensorParamNetConfig):
        super().__init__()

        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[config.positivity_func]
        self.tpnet_config = config
        self.w = torch.nn.Linear(
            config.in_dim, config.out_dim_encoder * config.hidden_dim
        )
        self.decoder = (
            nn.Linear(config.hidden_dim, config.out_dim_decoder)
            if config.use_decoder
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D)

        Returns:
            torch.Tensor: Output tensor of shape (*, `out_dim_encoder x out_dim_decoder`)
        """
        params = self.w(x)  # (*, D) => (*, `out_dim_encoder x hidden_dim`)
        params = params.reshape(
            *params.size()[:-1],
            self.tpnet_config.out_dim_encoder,
            self.tpnet_config.hidden_dim
        )  # (*, `out_dim_encoder`, `hidden_dim`)
        if self.decoder:
            params = self.decoder(
                params
            )  # (*, `out_dim_encoder`, `hidden_dim`) => (*, `out_dim_encoder,  out_dim_decoder`)  i.e. (BT, RH, V)
        return self.positivity_func(params)
