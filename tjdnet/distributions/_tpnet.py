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
    positivity_func: Literal[
        "sq", "abs", "exp", "safe_exp", "sigmoid", "none", "relu", "leaky_relu"
    ] = "sigmoid"
    use_decoder: bool = True
    use_bias_encoder: bool = True
    use_bias_decoder: bool = True

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return asdict(self)


def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Safe exponential function to avoid overflow."""
    return torch.exp(torch.clamp(x, max=20.0))  # Clamp to avoid overflow


class TensorParamNet(nn.Module):
    def __init__(self, config: TensorParamNetConfig):
        super().__init__()

        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
            "safe_exp": safe_exp,
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "leaky_relu": torch.nn.functional.leaky_relu,
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
        self._check_nan_inf(x, "input")

        # e.g., (B, D) -> (B, RH, d)
        params = self.w(x)  # (*, D) => (*, `out_dim_encoder x hidden_dim`)
        self._check_nan_inf(
            params,
            "encoder",
            context={"input": x, "weight": self.w.weight, "bias": self.w.bias},
        )

        params = params.reshape(
            *params.size()[:-1],
            self.tpnet_config.out_dim_encoder,
            self.tpnet_config.hidden_dim,
        )  # (*, `out_dim_encoder`, `hidden_dim`)

        if self.decoder:
            params_before_decoder = params
            params = self.decoder(
                params
            )  # (*, `out_dim_encoder`, `hidden_dim`) => (*, `out_dim_encoder,  out_dim_decoder`)  i.e. (BT, RH, V)
            self._check_nan_inf(
                params,
                "decoder",
                context={
                    "input": params_before_decoder,
                    "weight": self.decoder.weight,
                    "bias": self.decoder.bias,
                },
            )

        # Apply positivity function
        params_before_positivity = params
        output = self.positivity_func(params)
        self._check_nan_inf(
            output,
            "positivity",
            context={
                "input": params_before_positivity,
                "func": self.tpnet_config.positivity_func,
                # "decoder.weight": self.decoder.weight if self.decoder else None,
                # "decoder.bias": self.decoder.bias if self.decoder else None,
            },
        )

        return output

    def _check_nan_inf(
        self, tensor: torch.Tensor, stage: str, context: Optional[dict] = None
    ):
        """Simple NaN/Inf check with minimal diagnostics."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self._describe_tensor(tensor, stage)

            # Show all context tensors/values
            if context:
                for key, value in context.items():
                    if isinstance(value, torch.Tensor):
                        self._describe_tensor(value, f"{stage}_{key}")
                    else:
                        print(f"{stage}_{key}: {value}")

            raise ValueError(f"TensorParamNet: NaN/Inf detected at {stage} stage")

    def _describe_tensor(self, tensor: torch.Tensor, name: str):
        """Print essential tensor statistics for debugging."""
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        print(
            f"TPNet - {name}: {tensor.shape}, min={tensor.min().item():.3f}, max={tensor.max().item():.3f}, "
            f"NaN={nan_count}/{tensor.numel()}, Inf={inf_count}/{tensor.numel()}"
        )
