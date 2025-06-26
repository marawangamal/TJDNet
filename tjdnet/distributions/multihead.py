from typing import Callable, Optional
import torch
import torch.nn as nn

from tjdnet.distributions._base import AbstractDist, BaseDistFromLinearConfig
from tjdnet.distributions._tjdist import BaseDistConfig


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
        """Compute cross-entropy loss for each position."""
        total_loss = 0.0
        for h in range(self.horizon):
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

# def forward(self, x: torch.Tensor, y: torch.Tensor):
#     """Computes loss for multi-head distribution with memory-efficient implementation.

#     This implementation follows the memory-efficient approach from the paper by
#     computing forward and backward passes sequentially for each head, accumulating
#     gradients at the trunk while freeing logits and their gradients after each head.

#     Args:
#         x (torch.Tensor): Input features. Shape (B, D).
#         y (torch.Tensor): Target labels. Shape (B, H).

#     Returns:
#         torch.Tensor: Computed loss. Shape (B,).
#     """
#     batch_size = x.size(0)
#     total_loss = 0.0

#     # Enable gradient computation for memory-efficient backward pass
#     x.requires_grad_(True)

#     for h in range(self.horizon):
#         # Forward pass through current head
#         logits = self.heads[h](x)  # (B, V)
#         loss = nn.CrossEntropyLoss(reduction="none")(logits, y[:, h])  # (B,)

#         # Backward pass for current head
#         loss.backward(retain_graph=(h < self.horizon - 1))

#         # Accumulate loss
#         total_loss += loss.detach()

#     # Free logits and their gradients to reduce memory usage
#     # This reduces peak GPU memory from O(nV + d) to O(V + d)
#     del logits

# return total_loss
