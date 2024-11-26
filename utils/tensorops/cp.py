from typing import List
import torch
import tensorly as tl

tl.set_backend("pytorch")


def select_from_cp_tensor(
    cp_params: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Selects an element from a CP tensor representation (batched).

    Args:
        cp_params (torch.Tensor): CP represention. Shape (B, R, T, D).
        indices (List[int]): Indices to select from the tensor. Shape (B, T).

    Returns:
        torch.Tensor: Selected elements of shape (B,)
    """
    batch_size, rank, seq_len, n_embd = cp_params.size()
    idx = indices.unsqueeze(1)  # (B, 1, T)
    idx = idx.repeat(1, rank, 1)  # (B, R, T)
    result = torch.gather(
        cp_params.reshape(-1, n_embd), dim=1, index=idx.reshape(-1, 1)
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


def materialize_cp_tensorV2(
    x: torch.Tensor,
):
    """Performs outer product of a tensor with itself.

    Note:
        B: Batch size
        H: Number of CP factors
        V: CP factor dimension
        R: CP rank

    Args:
        x (torch.Tensor): Tensor of shape (B, H, V, R)

    Returns:
        torch.Tensor: Tensor of shape (B, V**H)
    """

    B, H, V, R = x.size()
    contractions = []
    weights = torch.ones(R, device=x.device)
    for b in range(B):
        res = tl.cp_to_tensor(
            (weights, [x[b, h] for h in range(H)])
        )  # List of tensors of shape (V, R)
        contractions.append(res)
    return torch.stack(contractions, dim=0)  # (B, V, V, ..., V)  H times
