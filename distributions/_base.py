import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List


class BaseDistribution(ABC, torch.nn.Module):
    def __init__(self, horizon: int):
        """
        Abstract base class for distributions compatible with TJDGPT2.
        """
        super().__init__()
        self.horizon = horizon

    def _get_horizon(self, horizon: Optional[int]):
        horizon = self.horizon if horizon is None else horizon
        if horizon > self.horizon:
            raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
        return horizon

    @abstractmethod
    def get_norm_consts(
        self, last_hidden_state: torch.Tensor, horizon: int, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute the normalization constant for the distribution.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
            horizon (int): Number of steps to consider.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Normalization constant and additional scale factors.
        """
        pass

    @abstractmethod
    def evaluate_at_points(
        self, last_hidden_state: torch.Tensor, points: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
            points (torch.Tensor): Points to evaluate, of shape (B, H, D).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation results and additional scale factors.
        """
        pass

    @abstractmethod
    def generate(
        self, last_hidden_state: torch.Tensor, horizon: int, **kwargs
    ) -> torch.Tensor:
        """
        Generate sequences based on the distribution.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
            horizon (int): Number of steps to generate.

        Returns:
            torch.Tensor: Generated sequences of shape (B, H).
        """
        pass
