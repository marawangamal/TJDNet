from dataclasses import dataclass
from typing import Optional

import torch

from tjdnet.distributions._base import BaseDistFromLinearConfig, BaseDistribution
from tjdnet.distributions.tpnet import TensorParamNet, TensorParamNetConfig
from tjdnet.utils import sample_topk


@dataclass
class CPBDistConfig:
    rank: int
    horizon: int
    hidden_dim: int
    in_dim: int
    vocab_size: int


class CPBDist(BaseDistribution):
    def __init__(self, config: CPBDistConfig):
        self.rank = config.rank
        self.horizon = config.horizon
        self.vocab_size = config.vocab_size
        self.alpha_func = torch.nn.Linear(
            in_features=config.in_dim,
            out_features=config.rank,
        )
        self.param_func = TensorParamNet(
            config=TensorParamNetConfig(
                in_dim=config.in_dim,
                hidden_dim=config.hidden_dim,
                out_dim_encoder=config.rank * config.horizon,
                out_dim_decoder=config.vocab_size,
                positivity_func="none",
            )
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
        x = hidden_state
        p_tilde = self.param_func(x)  # (B, HR, V)
        y_out = torch.empty(
            x.size(0), 0, device=x.device, dtype=torch.long
        )  # (B, 0 -> H)
        probs_list = []
        for h in range(self.horizon):
            p_dists = torch.log_softmax(p_tilde, dim=-1).reshape(
                -1, self.horizon, self.rank, self.vocab_size
            )  # (B, H, R, V)
            py = (
                p_dists[:, :h]
                .gather(-1, y_out.reshape(-1, 1, 1).expand(-1, self.horizon, self.rank))
                .squeeze(-1)
            )  # (B, H, R)
            p_dist = p_dists[:, h]  # (B, R, V)
            log_p = torch.logsumexp(
                # (B, R, 1) + (B, R) + (B, R, V) -> (B, V)
                torch.log(self.alpha_func(x)).unsqueeze(-1)
                + py.sum(dim=1).unsqueeze(-1)
                + p_dist,
                dim=1,
            )
            probs = torch.exp(log_p)
            probs_list.append(probs)
            sample_fn = lambda x: (
                sample_topk(x, top_k=top_k)
                if do_sample
                else lambda x: sample_topk(x, top_k=1)
            )
            y = sample_fn(probs)  # (B, V) -> (B, 1)
            y_out = torch.cat([y_out, y.unsqueeze(1)], dim=1)

        return y_out, torch.stack(probs_list, dim=1)  # (B, H), (B, H, V)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        """Computes loss for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B,).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
        """
        alpha = self.alpha_func(x)  # (B, R)
        p_tilde = self.param_func(x)  # (B, HR, V)
        p_dists = torch.log_softmax(p_tilde, dim=-1)
        # h, r =
        # (B, HR, V) -> (B, HR)
        py = p_dists.gather(
            2, y.reshape(-1, 1, 1).expand(-1, self.horizon * self.rank, -1)
        ).squeeze(-1)

        log_py_prime = py.reshape(-1, self.horizon, self.rank).prod(dim=1)
        log_py = torch.logsumexp(
            torch.log(alpha) + log_py_prime, dim=1
        )  # (B, R) -> (B,)
        return -log_py
