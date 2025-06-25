from dataclasses import dataclass
from typing import Callable, Optional

import torch

from tjdnet.distributions._tjdist import (
    AbstractDistV2,
    BaseDistConfig,
    BaseDistFromLinearConfig,
)
from tjdnet.utils import sample_topk


@dataclass
class CPBDistConfig:
    rank: int
    horizon: int
    hidden_dim: int
    in_dim: int
    vocab_size: int


class CPBDist(AbstractDistV2):
    def __init__(self, config: BaseDistConfig):
        super().__init__()
        """Parameterized CP tensor network distribution.

        TN:
              α
            / |   \
           /  |    \
          θ₁  θ₂ .. θₕ
          |   |     |
          D   D     D
          |   |     |
          y₁  y₂ .. yₕ
        """

        # === dims
        self.vocab_size = config.vocab_size
        self.horizon = config.horizon
        self.rank = config.rank
        self.in_dim = config.param_net.in_dim
        self.hidden_dim = config.param_net.hidden_dim
        H, R, D, V = (self.horizon, self.rank, self.in_dim, self.vocab_size)

        # === params
        self.w_alpha = torch.nn.Linear(D, R)
        self.w_cp = torch.nn.Linear(D, R * H * V)
        self.decoder = torch.nn.Parameter(torch.randn(D, V))  # (d, V)

    @classmethod
    def from_pretrained(cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig):
        raise NotImplementedError(
            "from_linear method must be implemented in the subclass"
        )

    # This version uses a separate decoder
    def log_prob_v2(
        self,
        x: torch.Tensor,  # (B, D)
        y: torch.Tensor,  # (B, H)
        return_dists: bool = False,
        **kwargs,
    ):
        """Computes log probabilities for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).
            return_dists (bool, optional): Whether to return distribution p(y_H|x). Defaults to False.

        Returns:
            torch.Tensor: Computed log probabilities. Shape (B,).

        """

        # === dims
        H, R, D, V = (self.horizon, self.rank, self.in_dim, self.vocab_size)
        B = x.size(0)
        H_y = y.size(1)

        # === cp params
        theta = self.w_cp(x).reshape(-1, R, H, D)  # cores
        alpha = torch.softmax(self.w_alpha(x).reshape(-1, R), dim=-1)  # moe weights

        # === test mode
        if return_dists:
            if H_y == 0:
                pass
            # The contraction sums over R and D dimensions
            dec = self.decoder.reshape(1, 1, 1, D, V).expand(B, R, H_y, -1, -1)
            dec = dec.gather(-1, y.reshape(-1, 1, H_y, 1, 1).expand(-1, R, -1, D))
            cp_left = (dec * theta[:, :, :H_y]).sum(-1)  # (B, R, H_y)
            cp_left = cp_left * alpha.unsqueeze(-1)
            # (B, R, D) * (1, 1, D, V) => (B, R, D, V) => (B, R, V)
            cp_right = (theta[:, :, -1] * self.decoder.reshape(1, 1, D, V)).sum(-2)
            cp_right = cp_right * alpha.unsqueeze(-1)  # (B, R, V)
            p = (cp_left.prod(-1, keepdim=True) * cp_right).sum(1)  # (B, V)
            return p / p.sum(-1, keepdim=True)  # (B, V)

        # === train mode
        # (d, V) => (B, R, H, d, V) => (B, R, H, d)
        dec = self.decoder.reshape(1, 1, 1, D, V).expand(B, R, H, -1, -1)  # expand
        dec = dec.gather(-1, y.reshape(-1, 1, H, 1, 1).expand(-1, R, -1, D))  # select
        theta = (dec * theta).sum(dim=-1)  # (B, R, H)

        # (B, R, H) * (B, R, 1) => (B, R, H) => (B, R) => (B,)
        p = (theta * alpha.unsqueeze(-1)).prod(-1).sum(-1)
        return p.log()

    def log_prob(
        self,
        x: torch.Tensor,  # (B, D)
        y: torch.Tensor,  # (B, H)
        return_dists: bool = False,
        **kwargs,
    ):
        """Computes log probabilities for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).
            return_dists (bool, optional): Whether to return distribution p(y_H|x). Defaults to False.

        Returns:
            torch.Tensor: Computed log probabilities. Shape (B,).

        """

        # === dims
        H, R, D, V = (self.horizon, self.rank, self.in_dim, self.vocab_size)
        H_y = y.size(1)

        # === cp params
        theta = torch.softmax(self.w_cp(x).reshape(-1, R, H, V), dim=-1)  # cores
        alpha = torch.softmax(self.w_alpha(x).reshape(-1, R), dim=-1)  # moe weights

        # === test mode
        if return_dists:
            cp_left = None
            if H_y != 0:
                # The contraction sums over R and D dimensions
                # (B, R, H, V) => (B, R, H_y)
                cp_left = theta[:, :, :H_y].gather(
                    -1, y.reshape(-1, 1, H_y, 1).expand(-1, R, -1, -1)
                )
                cp_left = cp_left * alpha.unsqueeze(-1)

            # (B, R, D) * (1, 1, D, V) => (B, R, D, V) => (B, R, V)
            cp_right = theta[:, :, -1] * alpha.unsqueeze(-1)  # (B, R, V)
            p = cp_right  # (B, R, V)
            if cp_left is not None:
                p = cp_left.prod(-1, keepdim=True) * p  # (B, R, V)
            p = p.sum(1)  # (B, V)
            return p / p.sum(-1, keepdim=True)  # (B, V)

        # === train mode
        #  (B, R, H, V) => (B, R, H)
        theta_select = theta.gather(
            -1, y.reshape(-1, 1, H, 1).expand(-1, R, -1, -1)
        )  # select
        theta = theta_select.sum(dim=-1)  # (B, R, H)

        # (B, R, H) * (B, R, 1) => (B, R, H) => (B, R) => (B,)
        p = (theta * alpha.unsqueeze(-1)).prod(-1).sum(-1)
        return p.log()

    def sample(
        self,
        x: torch.Tensor,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int],
        return_logits: bool = False,
        **kwargs,
    ):
        """Computes loss for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
        """
        y_out = torch.empty(x.size(0), 0, device=x.device, dtype=torch.long)
        for _ in range(self.horizon):
            prob_y_bar_xy = self.log_prob(x, y_out, return_dists=True)
            y_out_t = sample_fn(prob_y_bar_xy).unsqueeze(1)  # (B, 1)
            y_out = torch.cat([y_out, y_out_t], dim=-1)  # (B, H+1)
        return y_out, prob_y_bar_xy

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes loss for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
        """
        return -self.log_prob(x, y)
