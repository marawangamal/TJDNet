"""Minimal Mixture of Experts (MOE) layer implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MOELayer(nn.Module):
    """Minimal Mixture of Experts layer.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for experts
        num_experts: Number of expert networks
        num_experts_per_token: Number of experts to route each token to
        router_type: Type of router ('top_k' or 'noisy_gate')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        router_type: str = "top_k",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.router_type = router_type
        self.dropout = dropout

        # Router (gate) network
        self.router = nn.Linear(input_dim, num_experts)

        # Expert networks
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor, return_router_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the MOE layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_router_logits: Whether to return router logits for analysis

        Returns:
            Tuple of (output, router_logits)
        """
        batch_size, seq_len, _ = x.shape

        # Get router logits
        router_logits = self.router(x)  # (batch_size, seq_len, num_experts)

        # Route tokens to experts
        if self.router_type == "top_k":
            expert_weights, expert_indices = self._top_k_routing(router_logits)
        else:
            raise ValueError(f"Unknown router type: {self.router_type}")

        # Process through experts
        output = torch.zeros_like(x)
        for i in range(self.num_experts_per_token):
            expert_idx = expert_indices[:, :, i]  # (batch_size, seq_len)
            expert_weight = expert_weights[:, :, i].unsqueeze(
                -1
            )  # (batch_size, seq_len, 1)

            # Create mask for this expert
            expert_mask = torch.arange(self.num_experts, device=x.device).unsqueeze(
                0
            ).unsqueeze(0) == expert_idx.unsqueeze(-1)

            # Apply expert to masked tokens
            for j, expert in enumerate(self.experts):
                mask = expert_mask[:, :, j]  # (batch_size, seq_len)
                if mask.any():
                    expert_output = expert(x[mask])  # (num_masked_tokens, input_dim)
                    output[mask] += expert_weight[mask] * expert_output

        if return_router_logits:
            return output, router_logits
        return output, None

    def _top_k_routing(
        self, router_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts.

        Args:
            router_logits: Logits from router network

        Returns:
            Tuple of (expert_weights, expert_indices)
        """
        # Get top-k experts for each token
        expert_weights, expert_indices = torch.topk(
            router_logits, k=self.num_experts_per_token, dim=-1
        )

        # Apply softmax to get probabilities
        expert_weights = F.softmax(expert_weights, dim=-1)

        return expert_weights, expert_indices
