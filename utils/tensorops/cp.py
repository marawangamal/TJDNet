import torch


def select_from_cp_tensor(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Selects the indices from the CP tensor.

    Args:
        tensor (torch.Tensor): CP represention of shape (B, R, T, D)
        indices (List[int]): Indices to select from the tensor of shape (B, T)

    Returns:
        torch.Tensor: Selected tensor of shape (B)
    """
    batch_size, rank, seq_len, n_embd = tensor.size()
    idx = indices.unsqueeze(1)  # (B, 1, T)
    idx = idx.repeat(1, rank, 1)  # (B, R, T)
    result = torch.gather(
        tensor.reshape(-1, n_embd), dim=1, index=idx.reshape(-1, 1)
    )  # (B * R * T, 1)
    # Now need to reshape back to (B, R, T)
    result = result.reshape(batch_size, rank, seq_len)
    return result.prod(dim=2).sum(dim=1)


def sum_cp_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Sums the CP tensor.

    Args:
        tensor (torch.Tensor): CP represention of shape (B, R, T, D)

    Returns:
        torch.Tensor: Summed tensor of shape (B)
    """
    _, _, seq_len, _ = tensor.size()
    result = None
    for t in range(seq_len):
        if result is None:
            result = tensor[:, :, t, :].sum(dim=-1)  # (B, R)
        else:
            result = (tensor[:, :, t, :] * result.unsqueeze(2)).sum(dim=-1)  # (B, R)

    if result is None:
        raise ValueError("Empty tensor")
    return result.sum(dim=1)  # (B, R) -> (B)
