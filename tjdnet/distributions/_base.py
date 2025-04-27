from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import torch

from tjdnet.distributions.tpnet import TensorParamNet, TensorParamNetConfig


@dataclass
class BaseDistConfig:
    """Configuration for base distribution models.

    This class defines the core parameters shared across different tensor network
    distribution implementations (CP, MPS, etc.).

    Attributes:
        # Core Distribution Parameters
        vocab_size (int): Size of the vocabulary for token generation.
        vocab_size_compr (int): Compressed vocabulary size.
        horizon (int): Number of future tokens to predict.
        rank (int): Rank of the tensor decomposition.
            - Higher rank allows modeling more complex token dependencies
            - But increases computation and memory requirements

        # Network Architecture
        param_net (TensorParamNetConfig): Configuration for the parameter network
            that transforms embeddings into distribution parameters.
    """

    vocab_size: int
    horizon: int
    rank: int
    param_net: TensorParamNetConfig
    vocab_size_compr: int = 4096


class BaseDistribution(ABC, torch.nn.Module):
    def __init__(self, config: BaseDistConfig):
        """Abstract base class for distributions compatible with TJDGPT2."""
        super().__init__()
        self.vocab_size = config.vocab_size
        self.vocab_size_compr = config.vocab_size_compr
        self.horizon = config.horizon
        self.rank = config.rank
        self.cache = {}
        self.param_func = TensorParamNet(config.param_net)

    def _get_horizon(self, horizon: Optional[int]):
        horizon = self.horizon if horizon is None else horizon
        if horizon > self.horizon:
            raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
        return horizon

    def _get_params_from_cache(
        self,
        last_hidden_state: torch.Tensor,
        use_cache: bool,
        save_cache: bool,
        **kwargs,
    ) -> torch.Tensor:
        params = None
        if use_cache and "hidden_state" in self.cache:
            params = self.cache["hidden_state"]
        else:
            params = self._get_params(last_hidden_state)
            if save_cache:
                self.cache["hidden_state"] = params
        return params

    def init_params(
        self, pt_weight: torch.Tensor, pt_bias: Optional[torch.Tensor]
    ) -> None:
        raise NotImplementedError("init_params method must be implemented")

    @abstractmethod
    def _get_params(self, last_hidden_state: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(
        self,
        hidden_state: torch.Tensor,
        horizon: Optional[int],
        do_sample: bool,
        top_k: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the distribution.

        Args:
            hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
            horizon (int): Number of future tokens to predict.

        Returns:
            tuple:
                - torch.Tensor: Sampled tokens of shape (B, H).
                - torch.Tensor: Probs of shape (B, H, V).
        """
        pass

    @abstractmethod
    def evaluate_at_points_and_get_norm_consts(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """Evaluate the distribution at the given points and get normalization constants.

        Computes p'(y) and z(y) for the given points y.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, H).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
                - Evaluation of the distribution at the points of shape (B, H).
                - Scale tensors (empty list).
                - Normalization constants of shape (B, T).
                - Scale tensors for normalization constants (empty list).
        """
        pass
