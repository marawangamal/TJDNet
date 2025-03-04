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


def sample_from_tensor_dist(tens, num_samples=1):
    """Sample from tensor representing a distribution.

    Args:
        tens (torch.Tensor): Joint distribution tensor.
        num_samples (int): Number of samples to draw.

    Returns:
        torch.Tensor: Sampled indices. Shape: (num_samples, ...).
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


def get_breakpoints(ops: torch.Tensor):
    """Get breakpoints for select, free, and marginalize operations.

    Args:
        ops (torch.Tensor): Operation codes of shape (B, T) specifying:
            -2: marginalize mode (sum reduction)
            -1: keep mode as free index
            [0,V): select index v in mode

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Breakpoints for select/free (h_slct) and free/marginalize (h_mrgn) operations

    """
    # For first non-select (first -1 or -2)
    non_select_mask = (ops < 0).int()  # Convert bool to int, shape: (B, T)
    has_non_select = non_select_mask.any(dim=1)
    h_free = non_select_mask.argmax(dim=1)  # shape: (B,)
    # For batches with all selects, set h_slct to T
    h_free = torch.where(
        has_non_select, h_free, torch.tensor(ops.size(1), device=ops.device)
    )

    # For first margin (first -2)
    is_margin_mask = (ops == -2).int()  # Convert bool to int
    has_margin = is_margin_mask.any(dim=1)
    h_mrgn = is_margin_mask.argmax(dim=1)
    h_mrgn = torch.where(
        has_margin, h_mrgn, torch.tensor(ops.size(1), device=ops.device)
    )
    return h_free.long(), h_mrgn.long()


def mps_to_tensor(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
):
    """Converts a MPS tensor representation to a tensor.

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (R)
        beta (torch.Tensor): Beta tensor of shape (R)
        core (torch.Tensor): Core tensor of shape (H, R, D, R)

    Returns:
        torch.Tensor: Tensor representation of shape (D,) * H
    """
    full_shape = [core.size(2)] * core.size(0)
    result = torch.einsum("r, rdj -> dj", alpha, core[0])
    for t in range(1, core.size(0)):
        result = torch.einsum("kr, rdj -> kdj", result, core[t])
        result = result.reshape(-1, core.size(1))
    result = torch.einsum("kr,r -> k", result, beta)
    return result.reshape(full_shape)
