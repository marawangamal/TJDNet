from typing import List, Tuple
import torch
import tensorly as tl

from tjdnet.tensorops.common import get_breakpoints

tl.set_backend("pytorch")


def select_margin_ccp_tensor_batched(
    cp_params: torch.Tensor,
    cp_decode: torch.Tensor,
    ops: torch.Tensor,
    use_scale_factors=False,
):
    """Performs selection and marginalization operations on a CP tensor representation.

    Given a CP tensor T = ∑ᵢ aᵢ₁ ⊗ aᵢ₂ ⊗ ... ⊗ aᵢₜ where each aᵢⱼ ∈ ℝᵈ,
    applies a sequence of operations on each mode:
        - Selection: For op ∈ [0,V), selects op^th index of aᵢⱼ
        - Marginalization: For op = -2, sums all elements in aᵢⱼ
        - Free index: For op = -1, keeps aᵢⱼ unchanged

    Args:
        cp_params (torch.Tensor): CP tensor factors of shape (B, R, T, d)
        cp_decode (torch.Tensor): CP decode factors of shape (B, d, D)
        ops (torch.Tensor): Operation codes of shape (B, T) specifying:
            -2: marginalize mode (sum reduction)
            -1: keep mode as free index
            [0,V): select index v in mode

    Note:
        - The number of free indices in `ops` must be at most 1

    Returns:
        result (torch.Tensor): Result tensor of shape (n_free, D).

        scale_factors (list): Scale factors list of shape (T,).

    """

    # Validation:
    assert len(cp_params.shape) == 4, "CP params tensor must be $D (batched)"
    assert len(ops.shape) == 2, "Invalid ops tensor: must be 2D (batched)"
    assert (ops >= -2).all() and (
        ops < cp_params.size(3)
    ).all(), "Invalid ops tensor: must be in range [-2, vocab_size)"
    assert ops.size(0) == cp_params.size(0), "Batch size mismatch"

    batch_size, rank, horizon, compressed_dim = cp_params.size()
    _, _, uncompressed_dim = cp_decode.size()

    # Get breakpoints for 1st free leg and 1st margin leg
    bp_free, bp_margin = get_breakpoints(ops)  # (batch_size,), (batch_size,)

    res_left = torch.ones(batch_size, rank, device=cp_params.device)
    res_right = torch.ones(batch_size, rank, device=cp_params.device)
    res_free = torch.ones(batch_size, rank, uncompressed_dim, device=cp_params.device)

    # (BT, R, d) @ (BT, d, 1) -> (BT, R, 1)
    core_margins = (
        torch.bmm(
            # (B, R, T d) => (BT, R, d)
            cp_params.permute(0, 2, 1, 3).reshape(-1, rank, compressed_dim),
            # (B, d, D) => (B, d, 1) => (B, 1, d, 1) => (B, H, d, 1)
            cp_decode.sum(dim=-1, keepdim=True)
            .unsqueeze(1)
            .expand(-1, horizon, -1, -1)
            .reshape(-1, compressed_dim, 1),
        )
        .reshape(batch_size, horizon, rank)
        .permute(0, 2, 1)
    )  # (B, R, T)

    decoded_margin = cp_decode.sum(dim=-1, keepdim=True)  # (B, d, 1)

    for t in range(horizon):
        mask_select = t < bp_free
        mask_margin = t >= bp_margin
        mask_free = ~mask_select & ~mask_margin

        # Select
        if mask_select.any():

            # Select the corresponding indices from cp_decode
            cp_decode_expanded = torch.gather(
                cp_decode[mask_select].reshape(
                    -1, uncompressed_dim
                ),  # (batch_size' * d, D)
                dim=-1,
                index=ops[mask_select, t]
                .reshape(-1, 1, 1)
                .repeat(1, compressed_dim, 1)
                .reshape(-1, 1),  # (batch_size' * d, D)
            ).reshape(
                -1, compressed_dim, 1
            )  # (batch_size', d, 1)

            # Reshape cp_params to (batch_size', rank, compressed_dim)
            res_left[mask_select] = res_left[mask_select] * (
                torch.bmm(
                    # (B', R, d) @ (B', d, 1) -> (B, R, 1)
                    # (B, R, T, d) => (B', R, d) =>
                    cp_params[mask_select, :, t, :],
                    cp_decode_expanded,
                )
            ).squeeze(-1)

        # Marginalize
        if mask_margin.any():
            res_right[mask_margin] = (
                res_right[mask_margin] * core_margins[mask_margin, :, t]
            )
            # (B', R) * (B', R)
            # res_right[mask_margin] = res_right[mask_margin] * torch.bmm(
            #     # (B', R, d) @ (B', d, 1) -> (B', R, 1)
            #     cp_params[mask_margin, :, t, :],  # (B', R, d)
            #     decoded_margin[mask_margin],
            # ).squeeze(-1)

        # Free
        if mask_free.any():
            # (B', R, d) @ (B', d, D) -> (B', R, D)
            res_free[mask_free] = torch.bmm(
                cp_params[mask_free, :, t, :],  # (B', R, d)
                cp_decode[mask_free],  # (B', d, D)
            )

    # Final result

    # Special case: no free or margin legs (pure select)
    if torch.all(bp_free == horizon):
        return res_left.sum(dim=-1), []  # (B,)

    result = res_left.unsqueeze(-1) * res_free * res_right.unsqueeze(-1)  # (B, R, D)
    return result.sum(dim=1), []
