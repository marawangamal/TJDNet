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
    def get_norm_consts(
        self, last_hidden_state: torch.Tensor, horizon: int, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute the normalization constant for the distribution.

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
        """Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
            points (torch.Tensor): Points to evaluate, of shape (B, H, D).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation results and additional scale factors.
        """
        pass

    # @abstractmethod
    # def get_dist(
    #     self,
    #     hidden_state: torch.Tensor,
    #     ops: torch.Tensor,
    #     use_cache: bool,
    #     save_cache: bool,
    # ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    #     """Get distribution specified by ops.

    #     Args:
    #         last_hidden_state (torch.Tensor): Last hidden state of the transformer of shape (D)
    #         ops (torch.Tensor): Operation codes of shape (T,) specifying:
    #             -2: marginalize mode (sum reduction)
    #             -1: keep mode as free index
    #             [0,V): select index v in mode

    #     Returns:
    #         Tuple[torch.Tensor, List[torch.Tensor]]:
    #             - Distribution specified by ops
    #             - List of scale factors
    #     """
    #     pass

    @abstractmethod
    def sample(
        self,
        hidden_state: torch.Tensor,
        horizon: Optional[int],
        do_sample: bool,
        top_k: int,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the distribution.

        Args:
            hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
            horizon (int): Number of future tokens to predict.

        Returns:
            torch.Tensor: Sampled tokens of shape (B, H).
        """
        pass

    # @abstractmethod
    # def get_dist_batched(
    #     self,
    #     hidden_state: torch.Tensor,
    #     ops: torch.Tensor,
    # ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    #     """Get distribution specified by ops.

    #     Args:
    #         last_hidden_state (torch.Tensor): Last hidden state of the transformer of shape (B, D)
    #         ops (torch.Tensor): Operation codes of shape (B, T) specifying:
    #             -2: marginalize mode (sum reduction)
    #             -1: keep mode as free index
    #             [0,V): select index v in mode

    #     Returns:
    #         Tuple[torch.Tensor, List[torch.Tensor]]:
    #             - Distribution specified by ops
    #             - List of scale factors
    #     """
    #     pass

    # @abstractmethod
    # def sample(
    #     self,
    #     last_hidden_state: torch.Tensor,
    #     horizon: int = 1,
    #     do_sample: bool = True,
    #     top_k: int = 5,
    #     **kwargs,
    # ) -> torch.Tensor:
    #     """Sample from the distribution.

    #     Args:
    #         last_hidden_state (torch.Tensor): Hidden states of shape (B, T, D).
    #         horizon (int): Number of future tokens to predict.

    #     Returns:
    #         torch.Tensor: Sampled tokens of shape (B, H).
    #     """
    #     batch_size = last_hidden_state.size(0)
    #     y_hat = torch.empty((batch_size, 0), dtype=torch.long)
    #     dvc = last_hidden_state.device
    #     for h in range(horizon):
    #         ops_tensor = torch.tensor(
    #             y_hat.reshape()
    #             + -1 * torch.ones(batch_size, 1, dtype=torch.long)  # free leg
    #             + -2 * torch.ones(batch_size, (horizon - h - 1), dtype=torch.long),
    #             device=dvc,
    #         )
    #         p_ops_tilde, _ = self.get_dist(last_hidden_state, ops_tensor)
    #         if do_sample:
    #             top_k_scores, top_k_indices = torch.topk(
    #                 p_ops_tilde, k=min(top_k, p_ops_tilde.size(0))
    #             )
    #             top_k_probs = torch.softmax(top_k_scores, dim=0)  # (top_k,)
    #             sampled_indices = torch.multinomial(top_k_probs, num_samples=1)  # (1,)
    #             next_token = top_k_indices[sampled_indices].item()
    #         else:
    #             # Greedy decoding
    #             next_token = torch.argmax(p_ops_tilde, dim=-1).to(dvc)  # (1,)
    #         y_hat.append(next_token)

    #     return torch.stack(y_hat, dim=0).unsqueeze(0)  # (1, H)
