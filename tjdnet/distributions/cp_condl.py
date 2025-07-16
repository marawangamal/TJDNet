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
              a
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
        self.w_cp = torch.nn.Linear(D, H * R * V)
        self.decoder = torch.nn.Parameter(torch.randn(D, V))  # (d, V)

    @classmethod
    def from_pretrained(cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig):
        raise NotImplementedError(
            "from_linear method must be implemented in the subclass"
        )

    def _compute_log_prob(
        self,
        alpha_tilde: torch.Tensor,
        p_dists_tilde: torch.Tensor,
        y: torch.Tensor,
        return_dist_slice: bool = False,
        **kwargs,
    ):
        B, H, R, V = (
            y.size(0),
            self.config.horizon,
            self.config.rank,
            self.config.vocab_size,
        )
        assert y.size(1) <= H, f"y > horizon"
        if return_dist_slice:
            raise NotImplementedError("return_dist_slice not implemented")

        # log-softmax of unnormalized mixture weights. Shape: (B, R)
        lsm_alpha = torch.log_softmax(alpha_tilde, dim=-1)  # (B, R)
        # log-softmax of unnormalized conditional probs. Shape: (B, H, R, V)
        lsm_cp_cores = torch.log_softmax(p_dists_tilde.reshape(B, H, R, V), dim=-1)

        # Update prior using intermediate tokens
        z = (
            lsm_alpha.unsqueeze(-1)  # (B, R, 1)
            +
            # (B, H, R, V) -> (B, H, R, 1) -> (B, R, 1)
            lsm_cp_cores.gather(-1, y.reshape(B, H, 1, 1).expand(-1, -1, R, -1)).sum(
                dim=1
            )
        ).squeeze(-1)

        # (B, R) -> (B,)
        return torch.logsumexp(z, dim=1)

    def _compute_log_prob_unstable(
        self,
        alpha_tilde: torch.Tensor,
        p_dists_tilde: torch.Tensor,
        y: torch.Tensor,
        return_dist_slice: bool = False,
        **kwargs,
    ):
        B, H, R, V = (
            y.size(0),
            self.config.horizon,
            self.config.rank,
            self.config.vocab_size,
        )
        assert y.size(1) <= H, f"y > horizon"
        if return_dist_slice:
            raise NotImplementedError("return_dist_slice not implemented")

        # log-softmax of unnormalized mixture weights. Shape: (B, R)
        sm_alpha = torch.softmax(alpha_tilde, dim=-1)  # (B, R)
        # log-softmax of unnormalized conditional probs. Shape: (B, H, R, V)
        sm_cp_cores = torch.softmax(p_dists_tilde.reshape(B, H, R, V), dim=-1)
        # (B, H, R, V) -> (B, H, R)
        sm_cp_cores_slct = sm_cp_cores.gather(
            -1, y.reshape(B, H, 1, 1).expand(-1, -1, R, -1)
        ).squeeze(-1)

        # res = sm_cp_cores_slct.prod(dim=1).sum(dim=1)
        res = 0.0
        for r in range(R):
            res += sm_cp_cores_slct[:, :, r].prod(dim=1) * sm_alpha[:, r]
        return torch.log(res)

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_dist_slice: bool = False,
        **kwargs,
    ):
        """Computes logP(y|x) for CPCondl distribution.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape: (B, H).
            return_logit (bool, optional): Whether to return dist over next token. Defaults to False.

        Returns:
            torch.Tensor: log(p(y|x)). Shape: (B,)

        """
        if return_dist_slice:
            raise NotImplementedError("return_dist_slice not implemented")
        return self._compute_log_prob(
            self.w_alpha(x),  # (B, R)
            self.w_cp(x),  # (B, H*R*V)
            y,  # (B, H)
        )

    def log_prob_dist(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
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
        B, H, R, V = (
            y.size(0),
            self.config.horizon,
            self.config.rank,
            self.config.vocab_size,
        )
        assert y.size(1) <= H, f"y > horizon"

        # log-softmax of unnormalized mixture weights. Shape: (B, R)
        lsm_alpha = torch.log_softmax(self.w_alpha(x), dim=-1)  # (B, R)
        # log-softmax of unnormalized conditional probs. Shape: (B, H, R, V)
        lsm_cp_cores = torch.log_softmax(self.w_cp(x).reshape(B, H, R, V), dim=-1)

        # Select op on first h-1 conditional probs
        # (B, H, R, V) -> (B, H, R, 1) -> (B, R, 1)
        H_prime = y.size(1)  # current horizon idx
        lsm_cp_cores_slct = (
            lsm_cp_cores[:, :H_prime]
            .gather(-1, y.reshape(B, H_prime, 1, 1).expand(-1, -1, R, -1))
            .sum(dim=1)
        )

        lsm_cp_core_free = lsm_cp_cores[:, -1]  # (B, R, V)

        # Update prior using intermediate tokens
        z = lsm_alpha.unsqueeze(-1) + lsm_cp_cores_slct + lsm_cp_core_free  # (B, R, V)

        # (B, R) -> (B,)
        return torch.logsumexp(z, dim=1)

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
        """Sample from CPCondl distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            sample_fn (Callable[[torch.Tensor], torch.Tensor]): Function to sample from the distribution.
            horizon (Optional[int], optional): Horizon. Defaults to None.
            return_logits (bool, optional): Whether to return logits. Defaults to False.

        Returns:
            tuple:
                - Sampled tokens. Shape (B, H).
                - Logits of shape (B, H, V).
        """

        B, H = (x.size(0), self.config.horizon)
        y_out = torch.empty(B, 0, dtype=torch.long, device=x.device)
        for h in range(H):
            logits = self.log_prob_dist(x, y_out)  # (B, V)
            log_z = (
                self.log_prob(x, y_out).unsqueeze(-1)
                if y_out.size(1) > 0
                else torch.zeros_like(logits)
            )
            logits = logits - log_z
            yh = sample_fn(logits.exp())  # (B,)
            y_out = torch.cat([y_out, yh.unsqueeze(1)], dim=1)
        return y_out, []


if __name__ == "__main__":
    H, R, D, V = 3, 2, 4, 5
    cp_condl = CPCondl(BaseDistConfig(horizon=H, rank=R, embedding_dim=D, vocab_size=V))
    x = torch.randn(2, D)
    y = torch.randint(0, V, (2, H))
    print(cp_condl.log_prob(x, y))
    print(cp_condl.sample(x, lambda logits: logits.argmax(dim=-1)))
