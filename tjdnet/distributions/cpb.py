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
        super().__init__(config)

        # cp core weights
        self.alpha_unnorm_func = torch.nn.Linear(
            in_features=config.param_net.in_dim,
            out_features=config.rank,
        )

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

    def sample(
        self,
        hidden_state: torch.Tensor,
        horizon: Optional[int],
        do_sample: bool,
        top_k: int,
        **kwargs,
    ):
        x = hidden_state[:, -1]  # (B, D)
        cores = self.param_func(x)  # (B, HR, V)
        y_out = torch.empty(
            x.size(0), 0, device=x.device, dtype=torch.long
        )  # (B, 0 -> H)
        probs_list = []
        lsm_alpha = torch.log_softmax(self.alpha_unnorm_func(x), dim=-1)  # (B, R)
        for h in range(self.horizon):
            log_p_dists = torch.log_softmax(cores, dim=-1).reshape(
                -1, self.horizon, self.rank, self.vocab_size
            )  # (B, H, R, V)
            log_ph_dist = log_p_dists[:, h]  # (B, R, V)

            history = lsm_alpha
            if h > 0:
                log_pys = (
                    log_p_dists[:, :h]
                    .gather(
                        -1, y_out.reshape(-1, h, 1, 1).expand(-1, -1, self.rank, -1)
                    )
                    .squeeze(-1)
                )  # (B, h-1, R)
                history += log_pys.sum(dim=1)

            log_ph = torch.logsumexp(
                # (B, R, 1) + (B, R, V) -> (B, V)
                # self.log_alpha_func(x).unsqueeze(-1)
                # + py.sum(dim=1).unsqueeze(-1)
                history.unsqueeze(-1) + log_ph_dist,
                dim=1,
            )
            probs = torch.exp(log_ph)
            probs_list.append(probs)
            sample_fn = lambda x: (
                sample_topk(x, top_k=top_k)
                if do_sample
                else lambda x: sample_topk(x, top_k=1)
            )
            y = sample_fn(probs)  # (B, V) -> (B, 1)
            y_out = torch.cat([y_out, y], dim=1)

        return y_out, torch.stack(probs_list, dim=1)  # (B, H), (B, H, V)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        """Computes loss for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
        """
        lsm_alpha = torch.log_softmax(self.alpha_unnorm_func(x), dim=-1)  # (B, R)
        p_tilde = self.param_func(x)  # (B, HR, V)
        log_p_dists = torch.log_softmax(p_tilde, dim=-1).reshape(
            -1, self.horizon, self.rank, self.vocab_size
        )
        # (B, H, R, V) -> (B, H, R, 1) -> (B, H, R)
        log_py_prime = log_p_dists.gather(
            -1, y.reshape(-1, self.horizon, 1, 1).expand(-1, -1, self.rank, -1)
        ).squeeze(-1)

        log_py_pprime = log_py_prime.reshape(-1, self.horizon, self.rank).sum(dim=1)
        log_py = torch.logsumexp(lsm_alpha + log_py_pprime, dim=1)  # (B, R) -> (B,)
        return -log_py
