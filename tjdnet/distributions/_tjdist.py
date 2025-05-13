from __future__ import annotations  # only needed on 3.10 and below

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List, Type, TypeVar

import torch

from tjdnet.distributions.tpnet import TensorParamNet, TensorParamNetConfig

T = TypeVar("T", bound="AbstractDist")


@dataclass
class BaseDistFromLinearConfig:
    horizon: int
    rank: int
    param_net: TensorParamNetConfig


@dataclass
class BaseDistConfig:
    vocab_size: int
    horizon: int
    rank: int
    param_net: TensorParamNetConfig


class AbstractDist(ABC, torch.nn.Module):
    def __init__(self, config: BaseDistConfig):
        """Abstract base class for distributions compatible with TJD."""
        super().__init__()
        self.vocab_size = config.vocab_size
        self.horizon = config.horizon
        self.rank = config.rank
        self.param_func = TensorParamNet(config.param_net)

    def get_horizon(self, horizon: Optional[int]):
        horizon = self.horizon if horizon is None else horizon
        if horizon > self.horizon:
            raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
        return horizon

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    def from_linear(
        cls: Type[T], linear: torch.nn.Linear, config: BaseDistFromLinearConfig
    ) -> T:
        """Initialize the distribution from a linear layer.

        Args:
            linear (torch.nn.Linear): Linear layer to initialize the distribution.

        Returns:
            T: An instance of the distribution class.
        """
        pass


class TJDist(AbstractDist):
    def __init__(self, config: BaseDistConfig):
        super().__init__(config)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        p_tilde, gammas_p, z_tilde, gammas_z = self.evaluate(x, y)

        #  === Health checks >>>
        assert not torch.isnan(p_tilde).any(), "p_tilde NaN"
        assert not torch.isnan(z_tilde).any(), "norm_const NaN"
        if len(gammas_p) == 0 and len(gammas_z) == 0:
            if (p_tilde > z_tilde).any():
                print("p_tilde >= norm_const")
                print("p_tilde:", p_tilde)
                print("norm_const:", z_tilde)

            if not (p_tilde <= z_tilde).all():
                print("p_tilde > norm_const")
            assert (p_tilde <= z_tilde).all(), "p_tilde <= norm_const"
        # <<< Health checks ===

        loss = (
            -torch.log(p_tilde)  # (B, T')
            + torch.log(z_tilde)  # (B, T')
            # Contraction Stability Scale Factors
            - sum([torch.log(z) for z in gammas_p])  # (B, T')
            + sum([torch.log(z) for z in gammas_z])
        )  # (B, T-H)
        return loss

    @abstractmethod
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """Computes P(yh|x, y1:h-1) for h in [1, H].

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
                - Evaluation of the distribution at the points of shape (B, H).
                - Scale tensors (empty list).
                - Normalization constants of shape (B, T).
                - Scale tensors for normalization constants (empty list).
        """
        raise NotImplementedError("evaluate method must be implemented in the subclass")

    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes P(yh|x, y1:h-1) for h in [1, H].

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            sample_fn (Callable): Sampling function.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
                - Evaluation of the distribution at the points of shape (B, H).
                - Scale tensors (empty list).
                - Normalization constants of shape (B, T).
                - Scale tensors for normalization constants (empty list).
        """
        pass
