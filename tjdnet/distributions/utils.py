import torch


def safe_exp(x: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
    """Safe exponential function to avoid overflow."""
    return torch.exp(torch.clamp(x, max=max_val))  # Clamp to avoid overflow
