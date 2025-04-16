from typing import List, Tuple
import torch
import tensorly as tl
import line_profiler

from tjdnet.tensorops.common import get_breakpoints

tl.set_backend("pytorch")


def select_margin_cp_tensor_batched(
    cp_params: torch.Tensor, ops: torch.Tensor, use_scale_factors=False
):
    """Performs selection and marginalization operations on a CP tensor representation.

    Given a CP tensor T = ∑ᵢ aᵢ₁ ⊗ aᵢ₂ ⊗ ... ⊗ aᵢₜ where each aᵢⱼ ∈ ℝᵈ,
    applies a sequence of operations on each mode:
        - Selection: For op ∈ [0,V), selects op^th index of aᵢⱼ
        - Marginalization: For op = -2, sums all elements in aᵢⱼ
        - Free index: For op = -1, keeps aᵢⱼ unchanged

    Args:
        cp_params (torch.Tensor): CP tensor factors of shape (B, R, T, D) where:
            B: Batch size
            R: CP rank
            T: number of tensor modes/dimensions
            D: dimension of each mode
        ops (torch.Tensor): Operation codes of shape (T,) specifying:
            -2: marginalize mode (sum reduction)
            -1: keep mode as free index
            [0,V): select index v in mode

    Note:
        - The number of free indices in `ops` must be at most 1

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Result tensor of shape (n_free, D) where n_free is the number of free indices (-1 operations) in ops
            - Scale factors list of shape (T,)
    """

    # Validation:
    assert len(cp_params.shape) == 4, "CP params tensor must be $D (batched)"
    assert len(ops.shape) == 2, "Invalid ops tensor: must be 2D (batched)"
    assert (ops >= -2).all() and (
        ops < cp_params.size(3)
    ).all(), "Invalid ops tensor: must be in range [-2, vocab_size)"
    assert ops.size(0) == cp_params.size(0), "Batch size mismatch"

    batch_size, rank, horizon, vocab_size = cp_params.size()

    # Get breakpoints for 1st free leg and 1st margin leg
    bp_free, bp_margin = get_breakpoints(ops)  # (batch_size,), (batch_size,)

    res_left = torch.ones(batch_size, rank, device=cp_params.device)
    res_right = torch.ones(batch_size, rank, device=cp_params.device)
    res_free = torch.ones(batch_size, rank, vocab_size, device=cp_params.device)

    core_margins = cp_params.sum(dim=-1)  # (B, R, T)
    scale_factors = []

    for t in range(horizon):
        mask_select = t < bp_free
        mask_margin = t >= bp_margin
        mask_free = ~mask_select & ~mask_margin

        # Select
        if mask_select.any():
            update = torch.gather(
                # (B, R, T, D) -> (B', R, D)
                cp_params[mask_select, :, t, :].reshape(-1, vocab_size),
                dim=-1,
                index=ops[mask_select, t]  # (batch_size',)
                .reshape(-1, 1, 1)
                .repeat(1, rank, 1)
                .reshape(-1, 1),
            ).reshape(
                -1, rank
            )  # (batch_size', R)
            sf = torch.ones(batch_size, device=cp_params.device)  # (B,)
            sf[mask_select] = torch.max(update, dim=-1)[0]  # (B',)
            # BUG: used to be scale_factors.append( sf[mask_select])
            scale_factors.append(sf)
            res_left[mask_select] = (
                res_left[mask_select] * update / sf[mask_select].unsqueeze(-1)
            )

        # Marginalize
        if mask_margin.any():
            update = core_margins[mask_margin, :, t]  # (B', R)
            sf = torch.ones(batch_size, device=cp_params.device)  # (B,)
            sf[mask_margin] = torch.max(update, dim=-1)[0]  # (B',)
            # BUG: used to be scale_factors.append( sf[mask_margin])
            scale_factors.append(sf)
            res_right[mask_margin] = (
                res_right[mask_margin] * update / sf[mask_margin].unsqueeze(-1)
            )

        # Free
        if mask_free.any():
            res_free[mask_free] = cp_params[mask_free, :, t, :]

    # Final result

    # Special case: pure select
    if torch.all(bp_free == horizon):
        return res_left.sum(dim=-1), scale_factors  # (B,)
    # Special case: pure marginalization
    elif torch.all(bp_margin == 0):
        return res_right.sum(dim=-1), scale_factors
    else:  # General case
        result = (
            res_left.unsqueeze(-1) * res_free * res_right.unsqueeze(-1)
        )  # (B, R, D)
        return result.sum(dim=1), scale_factors


# ------------------------------------------------------------------------
# Deprecated functions (select_margin_cp_tensor_batched generalizes these)
# ------------------------------------------------------------------------


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


def select_margin_cp_tensor(
    cp_params: torch.Tensor, ops: torch.Tensor, use_scale_factors=False
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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

    Note:
        - The number of free indices in `ops` must be at most 1

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Result tensor of shape (n_free, D) where n_free is the number of free indices (-1 operations) in ops
            - Scale factors list of shape (T,)
    """

    # Validation:
    assert len(cp_params.shape) == 3, "CP params tensor must be 3D (non-batched)"
    assert len(ops.shape) == 1, "Ops tensor must be 1D (non-batched)"
    assert (ops >= -2).all() and (
        ops < cp_params.size(2)
    ).all(), "Invalid ops tensor: must be in range [-2, vocab_size)"

    rank_size, seq_len, vocab_size = cp_params.size()

    # Note ops must be in the order of select, free, marginalize
    bp_free, bp_margin = get_breakpoints(
        ops.reshape(1, -1)
    )  # (1,), (1,) selects index at which selects end
    bp_free, bp_margin = int(bp_free.item()), int(bp_margin.item())
    assert bp_free < bp_margin, "Invalid ops tensor (select/marginalize order)"
    assert (
        bp_margin - bp_free == 1
    ), "Invalid ops tensor: at most one free index allowed"

    cp_params = cp_params
    cp_params_select = cp_params[:, :bp_free, :]  # (R, bp_select, D)
    cp_params_margin = cp_params[:, bp_margin:, :]  # (R, n_mrgn, D)
    cp_params_free = cp_params[:, bp_free:bp_margin, :]  # (R, n_free, D)

    # Reduce selectable factors
    result = torch.gather(
        cp_params_select.reshape(-1, vocab_size),  # (R*n_slct, D)
        dim=1,
        # BUG: does .repeat(rank, 1) work? Dont we have R*n_slct indices?
        index=ops[:bp_free].reshape(1, -1).repeat(rank_size, 1).reshape(-1, 1),
    ).reshape(rank_size, bp_free, 1)

    # Reduce margin factors
    # Select and marginalize
    if result.size(1) > 0 and cp_params_margin.size(1) > 0:
        result = (
            result.prod(1).reshape(rank_size, 1, 1) * cp_params_margin
        )  # (R, n_mrgn, D)
    # Marginalize only
    elif cp_params_margin.size(1) > 0:
        result = cp_params_margin  # (R, n_mrgn, D)
    # BUG: this summation might be incorrect - technically we need to sum all possible combinations
    result = result.prod(dim=1).sum(dim=-1)  # (R,)

    # Multiply free factors
    result = result.reshape(rank_size, 1, 1) * cp_params_free  # (R, n_free, D)
    return result.sum(dim=0).reshape(-1), []
