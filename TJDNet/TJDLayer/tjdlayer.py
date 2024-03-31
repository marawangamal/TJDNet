from typing import Any, Callable
import torch.nn as nn
import torch


class TJDLayer(nn.Module):
    def __init__(
        self, emb_size, rank: int = 32, vocab_size: int = 128, *args, **kwargs
    ):
        """Tensor Train Joint Distribution Layer"""
        # Define TT JD parameters
        self.emb_size = emb_size
        self.rank = rank
        self.vocab_size = vocab_size
        self.w = nn.Parameter(
            torch.randn(rank + rank + (2 * rank + vocab_size), emb_size)
        )

    def _get_preds(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        """Get predictions from TT JD parameters

        Args:
            alpha: [R,]
            beta: [R,]
            core: [R, d, R]

        Returns:
            preds: [output_size]

        """
        raise NotImplementedError

    def _get_probs(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        """Get probabilities from TT JD parameters

        Args:
            alpha: [R,]
            beta: [R,]
            core: [R, d, R]

        Returns:
            probs: [output_size * vocab_size]
        """
        raise NotImplementedError

    def predict(self, x: torch.Tensor, output_size: int = 100) -> torch.Tensor:
        """Predict next token

        Args:
            x: [T, B, d]

        Returns:
            preds: [output_size]

        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, output_size: int = 100) -> torch.Tensor:
        """Forward pass for TT JD layer

        Args:
            x: [T, B, d]

        Returns:
            probs: [output_size * vocab_size]

        """
        seq_len, batch_size, emb_size = x.shape
        model_params = torch.mean(
            self.w @ x.reshape(-1, x.shape[-1]), dim=-1
        )  # [R + R + (2R + D)]
        alpha, beta, core = (
            model_params[: self.rank],
            model_params[self.rank : 2 * self.rank],
            model_params[2 * self.rank :],
        )
        core = core.reshape(self.rank, self.vocab_size, self.rank)
        probs = self._get_probs(alpha, beta, core, output_size)
        return probs
