from __future__ import annotations  # only needed on 3.10 and below

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Tuple, Type, TypeVar

import torch


T = TypeVar("T", bound="AbstractDist")


@dataclass
class BaseDistFromLinearConfig:
    horizon: int
    rank: int
    # param_net: TensorParamNetConfig
    embedding_dim: int = 768
    positivity_func: Literal["sq", "abs", "exp", "safe_exp", "sigmoid", "none"] = "exp"


@dataclass
class BaseDistConfig:
    vocab_size: int
    horizon: int
    rank: int
    # TODO: rename to activation_func
    # param_net: TensorParamNetConfig
    # tp config
    embedding_dim: int = 768
    positivity_func: Literal["sq", "abs", "exp", "safe_exp", "sigmoid", "none"] = "exp"


# class AbstractDist(ABC, torch.nn.Module):
#     def __init__(self, config: BaseDistConfig):
#         """Abstract base class for distributions compatible with TJD."""
#         super().__init__()
#     self.vocab_size = config.vocab_size
#     self.horizon = config.horizon
#     self.rank = config.rank
#     self.param_func = TensorParamNet(config.param_net)

# def get_horizon(self, horizon: Optional[int]):
#     horizon = self.horizon if horizon is None else horizon
#     if horizon > self.horizon:
#         raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
#     return horizon

#     @abstractmethod
#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """Computes loss for CPB distribution.

#         Args:
#             x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
#             y (torch.Tensor): Target labels. Shape (B, H).

#         Returns:
#             torch.Tensor: Computed loss. Shape (B,).
#         """
#         pass

#     @classmethod
#     @abstractmethod
#     def from_pretrained(
#         cls: Type[T], linear: torch.nn.Linear, config: BaseDistFromLinearConfig
#     ) -> T:
#         """Initialize the distribution from a linear layer.

#         Args:
#             linear (torch.nn.Linear): Linear layer to initialize the distribution.

#         Returns:
#             T: An instance of the distribution class.
#         """
#         pass


class AbstractDist(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes loss for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls: Type[T], linear: torch.nn.Linear, config: BaseDistFromLinearConfig
    ) -> T:
        """Initialize the distribution from a linear layer.

        Args:
            linear (torch.nn.Linear): Linear layer to initialize the distribution.

        Returns:
            T: An instance of the distribution class.
        """
        pass

    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int],
        return_logits: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes P(yh|x, y1:h-1) for h in [1, H].

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            sample_fn (Callable): Sampling function.
            horizon (Optional[int]): Horizon for sampling. Must be <= self.horizon.
            return_logits (bool): Whether to return logits or probabilities.

        Returns:
            tuple:
                - Evaluation of the distribution at the points of shape (B, H).
                - Probabilities of shape (B, H, V) or logits of shape (B, H, V).
        """
        pass
