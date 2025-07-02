from typing import Callable, Optional

import torch

from tjdnet.distributions._base import (
    AbstractDist,
    BaseDistConfig,
    BaseDistFromLinearConfig,
)


class CPCondl(AbstractDist):
    def __init__(self, config: BaseDistConfig):
        super().__init__()
        """Parameterized CP tensor network distribution (log space version).

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
        self.w_cp = torch.nn.Linear(D, R * H * V)
        self.decoder = torch.nn.Parameter(torch.randn(D, V))  # (d, V)

    @classmethod
    def from_pretrained(cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig):
        raise NotImplementedError(
            "from_linear method must be implemented in the subclass"
        )

    # def log_prob_unstable(
    #     self,
    #     x: torch.Tensor,
    #     y: Optional[torch.Tensor] = None,
    #     return_dist_slice: bool = False,
    # ):
    #     """Computes log P(y1,y2,...,yh|x) for CPB distribution (unstable version).

    #     Args:
    #         x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
    #         y (Optional[torch.Tensor], optional): Target labels. Shape (B, H'). Defaults to None.
    #         return_dist (bool, optional): Whether to return slice of the dist.

    #     Note:
    #         - H' is in the range [0, H].
    #         - Return dist does not return a probability distribution over the next token.
    #     """

    #     assert y.size(1) <= self.horizon if y is not None else True, f"y > horizon"

    #     # Get cp params

    #     alpha = torch.softmax(self.alpha_unnorm_func(x), dim=-1)  # (B, R)
    #     p_dists = torch.softmax(self.param_func(x), dim=-1).reshape(
    #         -1, self.horizon, self.rank, self.vocab_size
    #     )  # (B, H, R, V)

    #     py = alpha.unsqueeze(-1)  # (B, R, 1)
    #     h_prime = y.size(1) if y is not None else 0

    #     # Update prior using intermediate tokens
    #     if h_prime > 1 and y is not None:
    #         py = py * (
    #             # (B, H, R, V) -> (B, H', R, V) -> (B, R)
    #             p_dists[:, : h_prime - 1]
    #             .gather(  # (B, H', R, V)
    #                 -1,
    #                 y[:, : h_prime - 1]
    #                 .reshape(-1, h_prime - 1, 1, 1)
    #                 .expand(-1, -1, self.rank, -1),
    #             )
    #             .prod(1)
    #         )

    #     # Update prior using last token
    #     if return_dist_slice:
    #         py = py * (
    #             # (B, R, V) -> (B, V)
    #             p_dists[:, h_prime - 1]
    #         )

    #     else:
    #         # y cannot be None
    #         assert y is not None, "y must be provided if return_dist is False"
    #         py = py * (
    #             # (B, H, R, V) -> (B, R, V) -> (B, R)
    #             p_dists[:, h_prime - 1]
    #             .gather(
    #                 -1,
    #                 y[:, h_prime - 1].reshape(-1, 1, 1).expand(-1, self.rank, -1),
    #             )
    #             .squeeze(-1)
    #         )

    #     return torch.log(py.sum(1))  # (B, R) -> (B,)

    def log_prob(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_dist_slice: bool = False,
        **kwargs,
    ):
        """Computes logP(y|x) for CPCondl distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).
            return_logit (bool, optional): Whether to return dist over next token. Defaults to False.

        Returns:
            torch.Tensor: Computed logP(y|x). Shape (B,) if return_dist_slice is False, else (B, V).

        """
        H, R, D, V = (
            self.config.horizon,
            self.config.rank,
            self.config.embedding_dim,
            self.config.vocab_size,
        )
        assert y.size(1) <= H if y is not None else True, f"y > horizon"

        # Get cp params
        lsm_alpha = torch.log_softmax(self.w_alpha(x), dim=-1)  # (B, R)
        p_dists_tilde = self.w_cp(x)  # Unnorm cond probs. Shape: (B, HR, V)

        # Referred to as `a_tilde` in notes
        log_p_dists = torch.log_softmax(p_dists_tilde, dim=-1).reshape(-1, H, R, V)
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
                    .expand(-1, -1, R, -1),
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
                -1, y[:, h_prime - 1].reshape(-1, 1, 1).expand(-1, R, -1)
            )
            return torch.logsumexp(z, dim=1).squeeze(-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
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
        x: torch.Tensor,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        **kwargs,
    ):
        """_summary_

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            sample_fn (Callable): Sampling function.
            horizon (Optional[int]): Horizon for sampling. Must be <= self.horizon.
            return_logits (bool): Whether to return logits or probabilities.

        Returns:
            tuple:
                - Evaluation of the distribution at the points of shape (B, H).
                - Probabilities of shape (B, H, V) or logits of shape (B, H, V).
        """
        H, R, D, V = (
            self.config.horizon,
            self.config.rank,
            self.config.embedding_dim,
            self.config.vocab_size,
        )
        y = None
        x = x[:, -1]  # (B, D)
        z = torch.zeros(x.size(0), 1, device=x.device)
        pys = []
        horizon = min(horizon, H) if horizon is not None else H
        for _ in range(horizon):

            # log p(y_h|x, y_1:h-1) [B, V]
            log_py = self.log_prob(x, y, return_dist_slice=True)

            log_pyh_bar_y = log_py - z
            pyh_bar_y = torch.exp(log_pyh_bar_y)
            y_h = sample_fn(pyh_bar_y).unsqueeze(1)  # (B, 1)
            y = y_h if y is None else torch.cat([y, y_h], dim=-1)
            # (B, V) -> (B, 1)
            z = z + log_pyh_bar_y.gather(-1, y_h)

            # Save dist
            pys.append(pyh_bar_y)

        if y is None:
            raise ValueError("Failed to sample from CPB distribution.")

        return y, torch.stack(pys, dim=1)  # (B, H), (B, H, V)

    # def sample_v2(
    #     self,
    #     hidden_state: torch.Tensor,
    #     horizon: Optional[int],
    #     do_sample: bool,
    #     top_k: int,
    #     **kwargs,
    # ):
    #     H, R, D, V = (
    #         self.config.horizon,
    #         self.config.rank,
    #         self.config.embedding_dim,
    #         self.config.vocab_size,
    #     )
    #     y = None
    #     x = hidden_state[:, -1]  # (B, D)
    #     pys = []
    #     for h in range(H):
    #         log_pyh_bar_y_tilde = self.log_prob(x, y, return_dist_slice=True)  # (B, V)
    #         pyh_bar_y_tilde = torch.exp(log_pyh_bar_y_tilde)
    #         pyh_bar_y = pyh_bar_y_tilde / pyh_bar_y_tilde.sum(-1, keepdim=True)
    #         y_h = (
    #             sample_topk(pyh_bar_y, top_k=top_k)
    #             if do_sample
    #             else sample_topk(pyh_bar_y, top_k=1)
    #         )  # (B, 1)
    #         y = y_h if y is None else torch.cat([y, y_h], dim=-1)

    #         # Save dist
    #         pys.append(pyh_bar_y)

    #     if y is None:
    #         raise ValueError("Failed to sample from CPB distribution.")

    #     return y, torch.stack(pys, dim=1)  # (B, H), (B, H, V)
