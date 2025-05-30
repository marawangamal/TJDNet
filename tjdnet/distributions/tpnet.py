from dataclasses import asdict, dataclass
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
        use_bias_decoder: Whether to use a decoder layer
        use_bias_encoder: Whether to use a bias in the encoder layer
    """

    in_dim: int = 768
    hidden_dim: int = 512
    out_dim_encoder: int = 1
    out_dim_decoder: int = 1
    positivity_func: Literal["sq", "abs", "exp", "none"] = "exp"
    use_decoder: bool = True
    use_bias_encoder: bool = True
    use_bias_decoder: bool = True

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return asdict(self)


class TensorParamNet(nn.Module):
    def __init__(self, config: TensorParamNetConfig):
        super().__init__()

        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
            "none": lambda x: x,
        }[config.positivity_func]
        self.tpnet_config = config
        self.w = torch.nn.Linear(
            config.in_dim,
            config.out_dim_encoder * config.hidden_dim,
            bias=config.use_bias_encoder,
        )
        self.decoder = (
            nn.Linear(
                config.hidden_dim,
                config.out_dim_decoder,
                bias=config.use_bias_decoder,
            )
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
        # e.g., (B, D) -> (B, RH, d)
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
