import torch
from abc import ABC, abstractmethod
from typing import Tuple, List


class BaseDistribution(ABC, torch.nn.Module):
    def __init__(self):
        """
        Abstract base class for distributions compatible with TJDGPT2.
        """
        super().__init__()

    @abstractmethod
    def get_norm_consts(
        self, last_hidden_state: torch.Tensor, horizon: int
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
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        is_normalized: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
            points (torch.Tensor): Points to evaluate, of shape (B, H, D).
            is_normalized (bool, optional): Whether the output should be normalized.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation results and additional scale factors.
        """
        pass

    @abstractmethod
    def generate(self, last_hidden_state: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Generate sequences based on the distribution.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
            horizon (int): Number of steps to generate.

        Returns:
            torch.Tensor: Generated sequences of shape (B, H).
        """
        pass
