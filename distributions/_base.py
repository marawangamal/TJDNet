import torch
from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, List, TypeVar, Union


ParamType = TypeVar("ParamType")


class BaseDistribution(ABC, torch.nn.Module):
    def __init__(self, horizon: int):
        """
        Abstract base class for distributions compatible with TJDGPT2.
        """
        super().__init__()
        self.horizon = horizon
        self.cache = {}

    def _get_horizon(self, horizon: Optional[int]):
        horizon = self.horizon if horizon is None else horizon
        if horizon > self.horizon:
            raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
        return horizon

    # TODO: should always just return a tensor
    def _get_params_from_cache(
        self, hidden_state: torch.Tensor, use_cache: bool, save_cache: bool, **kwargs
    ) -> torch.Tensor:
        params = None
        if use_cache and "hidden_state" in self.cache:
            params = self.cache["hidden_state"]
        else:
            # BUGFIX: Dont assume shape
            params = self._get_params(hidden_state.reshape(1, 1, -1))  # (1, 1, R, H, V)
            if save_cache:
                self.cache["hidden_state"] = params
        return params

    @abstractmethod
    def _get_params(self, last_hidden_state: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

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
    def get_dist(
        self,
        hidden_state: torch.Tensor,
        ops: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get distribution specified by ops.

        Args:
            last_hidden_state (torch.Tensor): Last hidden state of the transformer of shape (D)
            ops (torch.Tensor): Operation codes of shape (T,) specifying:
                -2: marginalize mode (sum reduction)
                -1: keep mode as free index
                [0,V): select index v in mode

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Distribution specified by ops
                - List of scale factors
        """
        pass
