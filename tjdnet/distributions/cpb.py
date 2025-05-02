from dataclasses import dataclass
from typing import Optional

import torch

from tjdnet.distributions._base import (
    BaseDistConfig,
    BaseDistFromLinearConfig,
    BaseDistribution,
)
from tjdnet.utils import sample_topk


@dataclass
class CPBDistConfig:
    rank: int
    horizon: int
    hidden_dim: int
    in_dim: int
    vocab_size: int


class CPBDist(BaseDistribution):
    def __init__(self, config: BaseDistConfig):
        self.rank = config.rank
        self.horizon = config.horizon
        self.vocab_size = config.vocab_size
        config.param_net.out_dim_encoder = config.rank * config.horizon
        config.param_net.out_dim_decoder = config.vocab_size
        config.param_net.positivity_func = "none"
        super().__init__(config)

        # === learnable alpha (moe weights)
        self.alpha_unnorm_func = torch.nn.Linear(
            in_features=config.param_net.in_dim,
            out_features=config.rank,
        )
        # === fixed alpha
        # self.alpha_unnorm_func = lambda x: torch.ones(1, self.rank, device=x.device)

    def _get_params(self, last_hidden_state: torch.Tensor, **kwargs):
        raise NotImplementedError("_get_params method must be implemented")

    def evaluate_at_points_and_get_norm_consts(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError(
            "evaluate_at_points_and_get_norm_consts method must be implemented"
        )

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig):
        raise NotImplementedError(
            "from_linear method must be implemented in the subclass"
        )

    def log_prob(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_dist: bool = False,
        **kwargs,
    ):
        """Computes logP(y|x) for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).
            return_logit (bool, optional): Whether to return dist over next token. Defaults to False.

        Returns:
            torch.Tensor: Computed logP(y|x). Shape (B,) if return_logit is False, else (B, H, V).

        """
        lsm_alpha = torch.log_softmax(self.alpha_unnorm_func(x), dim=-1)  # (B, R)
        p_tilde = self.param_func(x)  # (B, HR, V)

        # Referred to as `a_tilde` in notes
        log_p_dists = torch.log_softmax(p_tilde, dim=-1).reshape(
            -1, self.horizon, self.rank, self.vocab_size
        )
        # (B, H, R, V) -> (B, H, R, 1) -> (B, H, R)

        history = lsm_alpha.unsqueeze(-1)  # (B, R, 1)
        h_idx = 0
        if y is not None:
            h_idx = y.size(1)
            history = history + (
                # (B, H, R, V) -> (B, H, R, 1) -> (B, R, 1)
                log_p_dists[:, :h_idx]
                .gather(  # (B, H', R, V)
                    -1,
                    y.reshape(-1, h_idx, 1, 1).expand(-1, -1, self.rank, -1),
                )
                .sum(dim=1)
            )

        if return_dist:
            # (B, R, 1) -> (B, R, V) -> (B, V)
            history = history + log_p_dists[:, h_idx]
            return torch.logsumexp(history, dim=1)

        else:
            # (B, R, 1) -> (B, 1) -> (B,)
            return torch.logsumexp(history, dim=1).squeeze(-1)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        """Computes loss for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
        """
        return -self.log_prob(x, y)

    def sample(
        self,
        hidden_state: torch.Tensor,
        horizon: Optional[int],
        do_sample: bool,
        top_k: int,
        **kwargs,
    ):
        y = None
        x = hidden_state[:, -1]  # (B, D)
        z = torch.log(torch.ones(x.size(0), 1, device=x.device))
        pys = []
        for h in range(self.horizon):
            log_py_dist = self.log_prob(x, y, return_dist=True)  # (B, V)
            py = torch.exp(log_py_dist - z)
            y_h = (
                sample_topk(py, top_k=top_k) if do_sample else sample_topk(py, top_k=1)
            )  # (B, 1)
            y = y_h if y is None else torch.cat([y, y_h], dim=-1)
            # (B, V) -> (B, 1)
            z = z + log_py_dist.gather(-1, y_h)

            # Save dist
            pys.append(py)

        if y is None:
            raise ValueError("Failed to sample from CPB distribution.")

        return y, torch.stack(pys, dim=1)  # (B, H), (B, H, V)
