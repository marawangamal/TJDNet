from __future__ import annotations  # only needed on 3.10 and below

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Tuple, Type, TypeVar

import torch
from tjdnet.types import PositivityFuncType


T = TypeVar("T", bound="AbstractDist")


@dataclass
class BaseDistFromLinearConfig:
    horizon: int
    rank: int
    embedding_dim: int = 768
    positivity_func: PositivityFuncType = "exp"


@dataclass
class BaseDistConfig:
    vocab_size: int
    horizon: int
    rank: int
    embedding_dim: int = 768
    positivity_func: PositivityFuncType = "exp"


class AbstractDist(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes loss for distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
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
        """Sample from distribution P(yh|x, y1:h-1) for h in [1, H].

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
