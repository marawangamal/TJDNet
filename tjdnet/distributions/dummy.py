from typing import Callable, List, Optional
import torch
import torch.nn as nn

from tjdnet.distributions._base import AbstractDist, BaseDistConfig
from tjdnet.distributions._tjdist import BaseDistConfig


class DummyDist(AbstractDist):
    """Dummy distribution for testing."""

    def __init__(self, config: BaseDistConfig):
        super().__init__()
        self.w_proj = nn.Linear(config.embedding_dim, config.vocab_size)

    @classmethod
    def from_pretrained(cls, linear: torch.nn.Linear, config: BaseDistConfig):
        raise NotImplementedError(
            "from_linear method must be implemented in the subclass"
        )

    def sample(
        self,
        x: torch.Tensor,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        **kwargs
    ):
        raise NotImplementedError("Sample not implemented for dummy distribution")

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        logits = self.w_proj(x)  # (B, V)
        loss = nn.CrossEntropyLoss(reduction="none")(logits, y[:, 0])  # (B,)
        return loss.mean()
