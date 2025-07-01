from typing import Callable, List, Optional
import torch
import torch.nn as nn

from tjdnet.distributions._base import (
    AbstractDist,
    BaseDistFromLinearConfig,
    BaseDistConfig,
)


class MultiHeadDist(AbstractDist):
    """Simple multi-head distribution with independent linear heads for each position."""

    def __init__(self, config: BaseDistConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.horizon = config.horizon
        self.embedding_dim = config.embedding_dim

        # Separate linear heads for each position
        self.heads = nn.ModuleList(
            [
                nn.Linear(config.embedding_dim, config.vocab_size)
                for _ in range(config.horizon)
            ]
        )

    @classmethod
    def from_pretrained(cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig):
        raise NotImplementedError(
            "from_linear method must be implemented in the subclass"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Compute cross-entropy loss for each position, or for one random head if partial=True."""
        total_loss = 0.0
        for h in range(self.horizon):
            logits = self.heads[h](x)
            loss = nn.CrossEntropyLoss(reduction="none")(logits, y[:, h])
            total_loss += loss
        return total_loss

    def forward_partial(
        self, x: torch.Tensor, y: torch.Tensor, head_ids: Optional[List[int]] = None
    ):
        """Compute cross-entropy loss for each position, or for one random head if partial=True."""
        # heads = (
        #     [int(torch.randint(0, self.horizon, (1,)).item())]
        #     if partial
        #     else range(self.horizon)
        # )
        total_loss = 0.0
        head_indices = range(self.horizon) if head_ids is None else head_ids
        for h in head_indices:
            logits = self.heads[h](x)
            loss = nn.CrossEntropyLoss(reduction="none")(logits, y[:, h])
            total_loss += loss
        return total_loss

    def sample(
        self,
        x: torch.Tensor,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        **kwargs
    ):
        """Sample from multi-head distribution."""
        H = horizon if horizon is not None else self.horizon
        B = x.size(0)
        device = x.device

        y_out = torch.empty(B, 0, device=device, dtype=torch.long)

        for h in range(H):
            logits = self.heads[h](x)
            probs = torch.softmax(logits, dim=-1)
            next_token = sample_fn(probs).unsqueeze(1)
            y_out = torch.cat([y_out, next_token], dim=1)
        return y_out, None


#    def forward(self, x, y):
#         h = self.backbone(x)
#         logits = [head(h) for head in self.heads]  # List of (B, V)
#         logits = torch.stack(logits, dim=1)  # (B, H, V)
#         # Compute mean cross-entropy loss over all positions
#         loss = torch.stack(
#             [self.criterion(logits[:, i, :], y[:, i]) for i in range(logits.size(1))]
#         ).mean()
#         return loss

#     def sample(self, x):
#         h = self.backbone(x)
#         logits = [head(h) for head in self.heads]
#         logits = torch.stack(logits, dim=1)
#         return torch.argmax(logits, dim=-1)
