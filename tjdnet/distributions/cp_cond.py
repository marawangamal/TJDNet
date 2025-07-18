from dataclasses import dataclass
from typing import Callable, Optional

import torch

from tjdnet.distributions._base import AbstractDist, BaseDistConfig


class CPCond(AbstractDist):
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
        self.config = config
        H, R, D, V = (
            self.config.horizon,
            self.config.rank,
            self.config.embedding_dim,
            self.config.vocab_size,
        )

        # === params
        self.w_alpha = torch.nn.Linear(D, R)
        self.w_cp_fac = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(D, H, R, D))
        )
        self.decoder = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(D, V))
        )

    def w_cp(self, x: torch.Tensor):
        return torch.einsum("be,ehrd,dv->brhv", x, self.w_cp_fac, self.decoder)

    @classmethod
    def from_pretrained(cls, linear: torch.nn.Linear, config: BaseDistConfig, **kwargs):
        """Create a CP distribution from a linear layer.

        Args:
            linear (torch.nn.Linear): Linear layer to use as a base. Shape: (D, V)
            config (BaseDistFromLinearConfig): Configuration for the distribution.

        Returns:
            CPDist: CP distribution with the given configuration.
        """

        vocab_size, embedding_dim = linear.weight.shape

        obj = cls(
            config=BaseDistConfig(
                vocab_size=vocab_size,
                horizon=config.horizon,
                rank=config.rank,
                embedding_dim=embedding_dim,
            ),
            **kwargs,
        )

        # Initialize the parameters in obj.tensor_param_net
        # with the parameters from the linear layer
        obj.decoder.weight.data = linear.weight.data  # type: ignore
        return obj

    def prob_y_bar_x(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ):

        # === dims
        H, R, D, V = (
            self.config.horizon,
            self.config.rank,
            self.config.embedding_dim,
            self.config.vocab_size,
        )
        H_y = y.size(1)

        # === cp params
        theta = torch.softmax(self.w_cp(x).reshape(-1, R, H, V), dim=-1)  # cores
        # w_cp = torch.einsum(
        #     "rhde,vd->rhdv", self.w_cp.weight.reshape(R, H, D, D), self.decoder.weight
        # )
        # theta = torch.softmax(torch.einsum("be,rhdv->brhv", x, w_cp), dim=-1)  # cores
        alpha = torch.softmax(self.w_alpha(x).reshape(-1, R), dim=-1)  # moe weights

        # === test mode - compute conditional distribution p(y_next | x, y_prev)
        if H_y == 0:
            # No previous tokens, use first core
            p = theta[:, :, 0] * alpha.unsqueeze(-1)  # (B, R, V)
        else:
            # Compute probability of next token given previous tokens
            # Get the core for the next position (H_y)
            next_core = theta[:, :, H_y]  # (B, R, V)

            # Get probabilities of previous tokens
            prev_probs = (
                theta[:, :, :H_y]
                .gather(-1, y.reshape(-1, 1, H_y, 1).expand(-1, R, -1, -1))
                .squeeze(-1)
            )  # (B, R, H_y)

            # Weight by alpha and multiply with next core
            p = (
                prev_probs.prod(-1, keepdim=True) * alpha.unsqueeze(-1)
            ) * next_core  # (B, R, V)

        # Sum over rank dimension and normalize
        p = p.sum(1)  # (B, V)
        return p / p.sum(-1, keepdim=True)  # (B, V)

    def log_prob(
        self,
        x: torch.Tensor,  # (B, D)
        y: torch.Tensor,  # (B, H)
        **kwargs,
    ):
        """Computes logP(y|x) for CPCond distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).
            return_dists (bool, optional): Whether to return distribution p(y_H|x). Defaults to False.

        Returns:
            torch.Tensor: Computed log probabilities. Shape (B,).

        """

        # === dims
        H, R, D, V = (
            self.config.horizon,
            self.config.rank,
            self.config.embedding_dim,
            self.config.vocab_size,
        )
        H_y = y.size(1)

        # === cp params
        theta = torch.softmax(self.w_cp(x).reshape(-1, R, H, V), dim=-1)  # cores
        # w_cp = torch.einsum(
        #     "rhde,vd->rhdv", self.w_cp.weight.reshape(R, H, D, D), self.decoder.weight
        # )
        # theta = torch.softmax(torch.einsum("be,rhdv->brhv", x, w_cp), dim=-1)  # cores
        alpha = torch.softmax(self.w_alpha(x).reshape(-1, R), dim=-1)  # moe weights

        # === train mode - evaluate joint probability p(y | x)
        #  (B, R, H, V) => (B, R, H)
        theta_select = theta.gather(
            -1, y.reshape(-1, 1, H, 1).expand(-1, R, -1, -1)
        ).squeeze(-1)

        # (B, R, H) * (B, R, 1) => (B, R, H) => (B, R) => (B,)
        p = (theta_select * alpha.unsqueeze(-1)).prod(-1).sum(-1)
        return p.log()

    def sample(
        self,
        x: torch.Tensor,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        **kwargs,
    ):
        """Sample from CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            sample_fn (Callable): Sampling function that takes probabilities and returns sampled indices.
            horizon (Optional[int]): Horizon for sampling. If None, uses self.horizon.
            return_logits (bool): Whether to return logits or probabilities.

        Returns:
            tuple: (sampled_sequence, final_probabilities)
                - sampled_sequence: Shape (B, H)
                - final_probabilities: Shape (B, V) - probabilities for the last token
        """
        y_out = torch.empty(x.size(0), 0, device=x.device, dtype=torch.long)
        horizon_to_use = horizon if horizon is not None else self.config.horizon

        for _ in range(horizon_to_use):
            prob_y_bar_xy = self.prob_y_bar_x(x, y_out)
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


if __name__ == "__main__":

    B, R, H, D, V = 1, 4, 2, 512, 1000

    config = BaseDistConfig(horizon=H, rank=R, embedding_dim=D, vocab_size=V)
    dist = CPCond(config)

    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    print(dist(x, y))
    print(dist.sample(x, sample_fn=lambda x: x.argmax(-1)))
