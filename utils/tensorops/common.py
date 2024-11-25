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
    assert (
        len(tens.shape) == indices.shape[1] + 1
    ), "Invalid shapes encountered in batch_multi_dim_index."

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


# TODO: rename to `sample_from_tensor_dist`
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
        samples = torch.unravel_index(samples, tens.shape)
        samples = torch.stack(samples, dim=-1)

    return samples


def get_windowed_input_ids(input_ids: torch.Tensor, horizon: int):
    # 1. Window the `input_ids` to get targets: (B, T) => (B, T, H)
    #   each position should look H steps ahead
    input_ids_windowed = window_input_ids(input_ids, horizon=horizon)

    # 2. Make targets using windowed input_ids
    targets = input_ids_windowed[:, :-horizon]  # (B, T-H, H)
    targets = targets.reshape(-1, horizon)  # (B * (T-H), H)
    return targets


def window_input_ids(input_ids: torch.Tensor, horizon: int, shift: int = 1):
    """Window the input_ids so that each position looks H steps ahead.

    Args:
        input_ids (torch.Tensor): The input tensor of shape (B, T).
        H (int): The number of steps ahead each position should look.

    Returns:
        torch.Tensor: The windowed tensor of shape (B, T, H).
    """
    B, T = input_ids.shape

    # Create the windowed input tensor
    input_ids_windowed = torch.stack(
        [torch.roll(input_ids, -i - shift, dims=1) for i in range(horizon)], dim=-1
    )

    # Mask out positions that roll beyond the sequence length
    for i in range(1, horizon):
        input_ids_windowed[:, -i - shift :, i] = (
            0  # Replace 0 with padding token if needed
        )

    # Correct the padding (zeroing) for positions that have rolled beyond the valid sequence length
    for i in range(horizon):
        # Calculate the index from which zeroing should start based on the shift
        zero_start = T - i - shift
        if zero_start < T:  # Only apply zeroing if we're within valid range
            input_ids_windowed[:, zero_start:, i] = 0

    return input_ids_windowed
