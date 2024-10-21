import torch
import numpy as np


def batch_multi_dim_index(
    tens: torch.Tensor,
    indices: torch.Tensor,
):
    """Perform a batched index select operation.
    Args:
        input (torch.Tensor): Input tensor. Shape: (B, d1, d2, ..., dN).
        indices (torch.Tensor): Batched index tensor. Shape: (B, N).
    Returns:
        torch.Tensor: Output tensor. Shape: (B,).
    """
    tens_shape = tens.size()
    tens_flat = tens.view(tens.size(0), -1)  # (B, d1 * d2 * ... * dN)
    indices_flat = torch.stack(
        [get_flat_index(indices[i], tens_shape[1:]) for i in range(tens.size(0))]
    )  # (B,)
    result = torch.gather(tens_flat, 1, indices_flat.unsqueeze(1)).squeeze(1)
    return result


def get_flat_index(indices, shape):
    """Convert multi-dimensional indices to a flat index.
    Args:
        indices (torch.Tensor): Multi-dimensional indices. Shape: (N,).
        shape (tuple): Shape of the tensor.
    Returns:
        int: Flat index.
    """
    return torch.tensor(np.ravel_multi_index(indices.cpu().numpy(), shape)).to(
        indices.device
    )


def sample_from_tens(tens, num_samples=1):
    """Sample from tensor representing a distribution.

    Args:
        tens (torch.Tensor): Joint distribution tensor.
        num_samples (int): Number of samples to draw.

    Returns:
        torch.Tensor: Sampled indices.
    """
    # Ensure the distribution is properly normalized
    tens = tens / tens.sum()

    # Flatten the distribution
    flat_dist = tens.view(-1)

    # Sample from the flattened distribution
    samples = torch.multinomial(flat_dist, num_samples, replacement=True)

    # If the original distribution was multi-dimensional, reshape the samples
    if len(tens.shape) > 1:
        samples = torch.unravel_index(samples, tens.shape[1:])
        samples = torch.stack(samples, dim=-1)

    return samples
