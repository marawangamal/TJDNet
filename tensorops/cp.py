from typing import List, Tuple, Union
import torch
import tensorly as tl
import line_profiler

from tensorops.common import get_breakpoints

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
    return result.prod(dim=2).sum(dim=1)  # (B,)


# @line_profiler.profile
def sum_cp_tensor(cp_params: torch.Tensor) -> torch.Tensor:
    """Sum all elements of a CP tensor representation (batched).

    Args:
        tensor (torch.Tensor): CP represention of shape (B, R, T, D)

    Returns:
        torch.Tensor: Summed tensor of shape (B)
    """
    _, _, seq_len, _ = cp_params.size()
    result = None
    for t in range(seq_len):
        if result is None:
            result = cp_params[:, :, t, :].sum(dim=-1)  # (B, R)
        else:
            result = (cp_params[:, :, t, :] * result.unsqueeze(2)).sum(dim=-1)  # (B, R)
    if result is None:
        raise ValueError("Empty tensor")
    return result.sum(dim=1)  # (B, R) -> (B)


def materialize_cp_tensor(
    x: torch.Tensor,
):
    """Performs outer product of a tensor with itself.

    Args:
        x (torch.Tensor): Tensor of shape (B, R, H, V)

    Returns:
        torch.Tensor: Tensor of shape (B, V**H)

    Note:
        B: Batch size
        H: Number of CP factors
        V: CP factor dimension
        R: CP rank

    """

    # (B, 1, R, H, V)
    B, R, H, V = x.size()
    contractions = []
    weights = torch.ones(R, device=x.device)
    for b in range(B):
        res = tl.cp_to_tensor(
            (weights, [x[b, :, h].T for h in range(H)])
        )  # List of tensors of shape (V, R)
        contractions.append(res)
    return torch.stack(contractions, dim=0)  # (B, V, V, ..., V)  H times


def sample_from_cp_tensor(cp_params: torch.Tensor) -> torch.Tensor:
    """Samples from a CP tensor representation of probabilities.

    Args:
        cp_params (torch.Tensor): CP tensor factors of shape (R, T, D) where:

    Raises:
        NotImplementedError: _description_

    Returns:
        torch.Tensor: Sampled tensor of shape (T,)
    """
    selected_indices = []
    for t in range(cp_params.size(1)):
        # Unnormalized P(y_t | y_{<t})
        # BUG: should not return rank dimensions
        p_tilde_yt_given_prev, _ = select_margin_cp_tensor(
            cp_params,
            ops=torch.tensor(
                selected_indices + [-1] + [-2] * (cp_params.size(1) - t - 1),
                device=cp_params.device,
            ),
        )  # (R, 1, D)
        # Sample from P(y_t | y_{<t})
        selected_index = torch.multinomial(
            p_tilde_yt_given_prev.sum(dim=0).reshape(-1), num_samples=1
        ).item()
        selected_indices.append(selected_index)
    return torch.tensor(selected_indices, device=cp_params.device)


def select_margin_cp_tensor(
    cp_params: torch.Tensor, ops: torch.Tensor, apply_scale_factors=False
) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
    """Performs selection and marginalization operations on a CP tensor representation.

    Given a CP tensor T = ∑ᵢ aᵢ₁ ⊗ aᵢ₂ ⊗ ... ⊗ aᵢₜ where each aᵢⱼ ∈ ℝᵈ,
    applies a sequence of operations on each mode:
        - Selection: For op ∈ [0,V), selects op^th index of aᵢⱼ
        - Marginalization: For op = -2, sums all elements in aᵢⱼ
        - Free index: For op = -1, keeps aᵢⱼ unchanged

    Args:
        cp_params (torch.Tensor): CP tensor factors of shape (R, T, D) where:
            R: CP rank
            T: number of tensor modes/dimensions
            D: dimension of each mode
        ops (torch.Tensor): Operation codes of shape (T,) specifying:
            -2: marginalize mode (sum reduction)
            -1: keep mode as free index
            [0,V): select index v in mode

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Result tensor of shape (R, F, D) where F is the numbe of free indices (-1 operations) in ops
            - Scale factors of shape (R,T)
    """

    # Validation:
    assert len(cp_params.shape) == 3, "CP params tensor must be 3D (non-batched)"
    assert len(ops.shape) == 1, "Ops tensor must be 1D (non-batched)"
    assert (ops >= -2).all() and (ops < cp_params.size(2)).all(), "Invalid ops tensor"

    # Note ops must be in the order of select, free, marginalize
    bp_free, bp_margin = get_breakpoints(
        ops.reshape(1, -1)
    )  # (1,), (1,) selects index at which selects end
    bp_free, bp_margin = int(bp_free.item()), int(bp_margin.item())
    assert bp_free < bp_margin, "Invalid ops tensor (select/marginalize order)"

    cp_params_scaled = cp_params
    scale_factors = None
    if apply_scale_factors:
        scale_factors = torch.linalg.norm(cp_params, dim=-1)  # (R, T)
        cp_params_scaled = cp_params / scale_factors.unsqueeze(-1)

    # Split CP tensor into selectable, margin, and free factors
    rank, seq_len, vocab_size = cp_params_scaled.size()
    cp_params_select = cp_params_scaled[:, :bp_free, :]  # (R, bp_select, D)
    cp_params_margin = cp_params_scaled[:, bp_margin:, :]  # (R, n_mrgn, D)
    cp_params_free = cp_params_scaled[:, bp_free:bp_margin, :]  # (R, n_free, D)

    # Reduce selectable factors
    result = torch.gather(
        cp_params_select.reshape(-1, vocab_size),  # (R*n_slct, D)
        dim=1,
        # BUG: does .repeat(rank, 1) work? Dont we have R*n_slct indices?
        index=ops[:bp_free].reshape(1, -1).repeat(rank, 1).reshape(-1, 1),
    ).reshape(rank, bp_free, 1)

    # Reduce margin factors
    if result.size(1) > 0 and cp_params_margin.size(1) > 0:
        result = result.prod(1).reshape(rank, 1, 1) * cp_params_margin  # (R, n_mrgn, D)
    elif cp_params_margin.size(1) > 0:
        result = cp_params_margin  # (R, n_mrgn, D)
    result = result.prod(dim=1).sum(dim=-1)  # (R,)

    # Multiply free factors
    result = result.reshape(rank, 1, 1) * cp_params_free  # (R, n_free, D)
    return result, scale_factors
