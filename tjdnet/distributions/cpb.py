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

    def log_prob_unstable(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_dist_slice: bool = False,
    ):
        """Computes log P(y1,y2,...,yh|x) for CPB distribution (unstable version).

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (Optional[torch.Tensor], optional): Target labels. Shape (B, H'). Defaults to None.
            return_dist (bool, optional): Whether to return slice of the dist.

        Note:
            - H' is in the range [0, H].
            - Return dist does not return a probability distribution over the next token.
        """

        assert y.size(1) <= self.horizon if y is not None else True, f"y > horizon"

        # Get cp params

        alpha = torch.softmax(self.alpha_unnorm_func(x), dim=-1)  # (B, R)
        p_dists = torch.softmax(self.param_func(x), dim=-1).reshape(
            -1, self.horizon, self.rank, self.vocab_size
        )  # (B, H, R, V)

        py = alpha.unsqueeze(-1)  # (B, R, 1)
        h_prime = y.size(1) if y is not None else 0

        # Update prior using intermediate tokens
        if h_prime > 1 and y is not None:
            py = py * (
                # (B, H, R, V) -> (B, H', R, V) -> (B, R)
                p_dists[:, : h_prime - 1]
                .gather(  # (B, H', R, V)
                    -1,
                    y[:, : h_prime - 1]
                    .reshape(-1, h_prime - 1, 1, 1)
                    .expand(-1, -1, self.rank, -1),
                )
                .prod(1)
            )

        # Update prior using last token
        if return_dist_slice:
            py = py * (
                # (B, R, V) -> (B, V)
                p_dists[:, h_prime - 1]
            )

        else:
            # y cannot be None
            assert y is not None, "y must be provided if return_dist is False"
            py = py * (
                # (B, H, R, V) -> (B, R, V) -> (B, R)
                p_dists[:, h_prime - 1]
                .gather(
                    -1,
                    y[:, h_prime - 1].reshape(-1, 1, 1).expand(-1, self.rank, -1),
                )
                .squeeze(-1)
            )

        return torch.log(py.sum(1))  # (B, R) -> (B,)

    def log_prob(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_dist_slice: bool = False,
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

        assert y.size(1) <= self.horizon if y is not None else True, f"y > horizon"

        # Get cp params
        lsm_alpha = torch.log_softmax(self.alpha_unnorm_func(x), dim=-1)  # (B, R)
        p_dists_tilde = self.param_func(x)  # Unnorm cond probs. Shape: (B, HR, V)

        # Referred to as `a_tilde` in notes
        log_p_dists = torch.log_softmax(p_dists_tilde, dim=-1).reshape(
            -1, self.horizon, self.rank, self.vocab_size
        )
        # (B, H, R, V) -> (B, H, R, 1) -> (B, H, R)

        z = lsm_alpha.unsqueeze(-1)  # (B, R, 1)
        h_prime = y.size(1) if y is not None else 0

        # Update prior using intermediate tokens
        if h_prime > 1 and y is not None:
            z = z + (
                # (B, H, R, V) -> (B, H', R, V) -> (B, H', R, 1) -> (B, R, 1)
                log_p_dists[:, : h_prime - 1]
                .gather(  # (B, H', R, V)
                    -1,
                    y[:, : h_prime - 1]
                    .reshape(-1, h_prime - 1, 1, 1)
                    .expand(-1, -1, self.rank, -1),
                )
                .sum(dim=1)
            )

        # Update prior using last token
        if return_dist_slice:
            z = z + (
                # (B, R, V) -> (B, V)
                log_p_dists[:, h_prime - 1]
            )
            return torch.logsumexp(z, dim=1)

        else:
            # (B, R, H, V) -> (B, R, 1)
            assert y is not None, "y must be provided if return_dist is False"
            z = z + log_p_dists[:, h_prime - 1].gather(
                -1, y[:, h_prime - 1].reshape(-1, 1, 1).expand(-1, self.rank, -1)
            )
            return torch.logsumexp(z, dim=1).squeeze(-1)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        """Computes loss for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
        """
        return -self.log_prob(x, y)

    def sample_v1(
        self,
        hidden_state: torch.Tensor,
        horizon: Optional[int],
        do_sample: bool,
        top_k: int,
        **kwargs,
    ):
        y = None
        x = hidden_state[:, -1]  # (B, D)
        z = torch.zeros(x.size(0), 1, device=x.device)
        pys = []
        for h in range(self.horizon):
            log_py = self.log_prob(x, y, return_dist_slice=True)  # (B, V)
            log_pyh_bar_y = log_py - z
            pyh_bar_y = torch.exp(log_pyh_bar_y)
            y_h = (
                sample_topk(pyh_bar_y, top_k=top_k)
                if do_sample
                else sample_topk(pyh_bar_y, top_k=1)
            )  # (B, 1)
            y = y_h if y is None else torch.cat([y, y_h], dim=-1)
            # (B, V) -> (B, 1)
            z = z + log_pyh_bar_y.gather(-1, y_h)

            # Save dist
            pys.append(pyh_bar_y)

        if y is None:
            raise ValueError("Failed to sample from CPB distribution.")

        return y, torch.stack(pys, dim=1)  # (B, H), (B, H, V)

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
        pys = []
        for h in range(self.horizon):
            log_pyh_bar_y_tilde = self.log_prob_unstable(
                x, y, return_dist_slice=True
            )  # (B, V)
            pyh_bar_y_tilde = torch.exp(log_pyh_bar_y_tilde)
            pyh_bar_y = pyh_bar_y_tilde / pyh_bar_y_tilde.sum(-1, keepdim=True)
            y_h = (
                sample_topk(pyh_bar_y, top_k=top_k)
                if do_sample
                else sample_topk(pyh_bar_y, top_k=1)
            )  # (B, 1)
            y = y_h if y is None else torch.cat([y, y_h], dim=-1)

            # Save dist
            pys.append(pyh_bar_y)

        if y is None:
            raise ValueError("Failed to sample from CPB distribution.")

        return y, torch.stack(pys, dim=1)  # (B, H), (B, H, V)
