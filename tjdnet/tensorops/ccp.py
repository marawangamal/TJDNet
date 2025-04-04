from typing import List, Tuple
import torch
import tensorly as tl


from tjdnet.tensorops.common import get_breakpoints

tl.set_backend("pytorch")


def select_margin_ccp_tensor_batched_v1(
    cp_params: torch.Tensor,
    cp_decode: torch.Tensor,
    ops: torch.Tensor,
    use_scale_factors=False,
):
    """Performs selection and marginalization operations on a CP tensor representation.

    Given a CP tensor Y = ∑ᵢ aᵢ₁ ⊗ aᵢ₂ ⊗ ... ⊗ aᵢₜ where each aᵢⱼ ∈ ℝᵈ,
    applies a sequence of operations on each mode:
        - Selection: For op ∈ [0,V), selects op^th index of aᵢⱼ
        - Marginalization: For op = -2, sums all elements in aᵢⱼ
        - Free index: For op = -1, keeps aᵢⱼ unchanged

    Args:
        cp_params (torch.Tensor): CP tensor factors of shape (B, R, H, d)
        cp_decode (torch.Tensor): CP decode factors of shape (B, d, D)
        ops (torch.Tensor): Operation codes of shape (B, H) specifying:
            -2: marginalize mode (sum reduction)
            -1: keep mode as free index
            [0,V): select index v in mode

    Note:
        - The number of free indices in `ops` must be at most 1

    Returns:
        result (torch.Tensor): Result tensor of shape (n_free, D).

        scale_factors (list): Scale factors list of shape (H,).

    """

    # Validation:
    assert len(cp_params.shape) == 4, "CP params tensor must be $D (batched)"
    assert len(ops.shape) == 2, "Invalid ops tensor: must be 2D (batched)"
    assert (ops >= -2).all() and (
        ops < cp_decode.size(2)
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
    # Largest intermediate tensor: (BTRD)
    core_margins = (
        torch.bmm(
            # (B, R, H d) => (BT, R, d)
            cp_params.permute(0, 2, 1, 3).reshape(-1, rank, compressed_dim),
            # (B, d, D) => (B, d, 1) => (B, 1, d, 1) => (B, H, d, 1)
            cp_decode.sum(dim=-1, keepdim=True)
            .unsqueeze(1)
            .expand(-1, horizon, -1, -1)
            .reshape(-1, compressed_dim, 1),
        )
        .reshape(batch_size, horizon, rank)
        .permute(0, 2, 1)
    )  # (B, R, H)

    # decoded_margin = cp_decode.sum(dim=-1, keepdim=True)  # (B, d, 1)

    for t in range(horizon):
        mask_select = t < bp_free
        mask_margin = t >= bp_margin
        mask_free = ~mask_select & ~mask_margin

        # Select
        if mask_select.any():

            # Select the corresponding indices from cp_decode
            cp_decode_expanded = torch.gather(
                # (B, d, D) => (B'd, D)
                cp_decode[mask_select].reshape(-1, uncompressed_dim),
                dim=-1,
                # (B, H) => (B',) => (B', 1, 1) => (B', d, 1) => (B'd, 1)
                index=ops[mask_select, t]
                .reshape(-1, 1, 1)
                .expand(-1, compressed_dim, -1)
                .reshape(-1, 1),  # (batch_size' * d, D)
            ).reshape(
                -1, compressed_dim, 1
            )  # (B', d, 1)

            # Reshape cp_params to (batch_size', rank, compressed_dim)
            # Largest intermediate tensor: (BRTD)
            res_left[mask_select] = res_left[mask_select] * (
                torch.bmm(
                    # (B', R, d) @ (B', d, 1) -> (B, R, 1)
                    # (B, R, H, d) => (B', R, d) =>
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


def select_margin_ccp_tensor_batched(
    cp_params: torch.Tensor,
    cp_decode: torch.Tensor,
    ops: torch.Tensor,
    use_scale_factors=False,
):
    """Performs selection and marginalization operations on a CP tensor representation.

    Given a CP tensor Y = ∑ᵢ aᵢ₁ ⊗ aᵢ₂ ⊗ ... ⊗ aᵢₜ where each aᵢⱼ ∈ ℝᵈ,
    applies a sequence of operations on each mode:
        - Selection: For op ∈ [0,V), selects op^th index of aᵢⱼ
        - Marginalization: For op = -2, sums all elements in aᵢⱼ
        - Free index: For op = -1, keeps aᵢⱼ unchanged

    Args:
        cp_params (torch.Tensor): CP tensor factors of shape (B, R, H, d)
        cp_decode (torch.Tensor): CP shared decoder of shape (d, D)
        ops (torch.Tensor): Operation codes of shape (B, H) specifying:
            -2: marginalize mode (sum reduction)
            -1: keep mode as free index
            [0,V): select index v in mode

    Note:
        - The number of free indices in `ops` must be at most 1
        - Scale factors should be multiplied (i.e. p = p_tilde * ∏i=1^k s_i)

    Returns:
        result (torch.Tensor): Result tensor of shape (n_free, D).
        scale_factors (list[torch.Tensor]): Scale factors list. Each scale factor is a tensor of shape (B,) for each marginalization step.

    """

    # Validation:
    assert len(cp_params.shape) == 4, "CP params tensor must be $D (batched)"
    assert len(ops.shape) == 2, "Invalid ops tensor: must be 2D (batched)"
    assert (ops >= -2).all() and (
        ops < cp_decode.size(1)
    ).all(), "Invalid ops tensor: must be in range [-2, vocab_size)"
    assert ops.size(0) == cp_params.size(0), "Batch size mismatch"

    batch_size, rank, horizon, compressed_dim = cp_params.size()
    _, uncompressed_dim = cp_decode.size()

    # Get breakpoints for 1st free leg and 1st margin leg
    bp_free, bp_margin = get_breakpoints(ops)  # (batch_size,), (batch_size,)

    res_left = torch.ones(batch_size, rank, device=cp_params.device)
    res_right = torch.ones(batch_size, rank, device=cp_params.device)
    res_free = torch.ones(batch_size, rank, uncompressed_dim, device=cp_params.device)

    # (BH, R, d) @ (BH, d, 1) -> (BH, R, 1)
    # Largest intermediate tensor: (BTRD)
    core_margins = (
        torch.bmm(
            # (B, R, H d) => (BH, R, d)
            cp_params.permute(0, 2, 1, 3).reshape(-1, rank, compressed_dim),
            # (d, D) => (d, 1) => (1, 1, d, 1) => (B, H, d, 1) => (BH, d, 1)
            cp_decode.sum(dim=-1, keepdim=True)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, horizon, -1, -1)
            .reshape(-1, compressed_dim, 1),
        )
        .reshape(batch_size, horizon, rank)
        .permute(0, 2, 1)
    )  # (B, R, H)

    # decoded_margin = cp_decode.sum(dim=-1, keepdim=True)  # (B, d, 1)
    scale_factors = []

    for t in range(horizon):
        mask_select = t < bp_free
        mask_margin = t >= bp_margin
        mask_free = ~mask_select & ~mask_margin

        # Select
        if mask_select.any():
            update = (
                # (B', R, d) @ (B', d, 1) -> (B, R, 1)
                torch.bmm(
                    # (B, R, H, d) => (B', R, d)
                    cp_params[mask_select, :, t, :],
                    # (d, D) => (d, B') => (B', d) => (B', d, 1)
                    cp_decode[:, ops[mask_select, t]].permute(1, 0).unsqueeze(-1),
                )
            ).squeeze(-1)
            sf = torch.ones(batch_size, device=cp_params.device)  # (B,)
            sf[mask_select] = torch.max(update, dim=-1)[0]  # (B',)
            res_left[mask_select] = (
                res_left[mask_select] * update / sf[mask_select].unsqueeze(-1)
            )
            scale_factors.append(sf)

        # Marginalize
        if mask_margin.any():
            update = core_margins[mask_margin, :, t]  # (B', R)
            # sf = torch.max(update, dim=-1)[0]  # (B',)
            sf = torch.ones(batch_size, device=cp_params.device)  # (B,)
            sf[mask_margin] = torch.max(update, dim=-1)[0]  # (B',)
            res_right[mask_margin] = (
                res_right[mask_margin] * update / sf[mask_margin].unsqueeze(-1)
            )
            scale_factors.append(sf)
            # (B', R) * (B', R)
            # res_right[mask_margin] = res_right[mask_margin] * torch.bmm(
            #     # (B', R, d) @ (B', d, 1) -> (B', R, 1)
            #     cp_params[mask_margin, :, t, :],  # (B', R, d)
            #     decoded_margin[mask_margin],
            # ).squeeze(-1)

        # Free
        if mask_free.any():
            # (B', R, d) @ (B', d, D) -> (B', R, D)
            b_prime = int(mask_free.sum().item())
            res_free[mask_free] = torch.bmm(
                cp_params[mask_free, :, t, :],  # (B', R, d)
                # cp_decode[mask_free],  # (B', d, D)
                cp_decode.unsqueeze(0).expand(b_prime, -1, -1),  # (B', d, D)
            )

    # Final result

    # Special case: pure select
    if torch.all(bp_free == horizon):
        return res_left.sum(dim=-1), scale_factors  # (B,)

    result = res_left.unsqueeze(-1) * res_free * res_right.unsqueeze(-1)  # (B, R, D)

    # Special case: pure marginalization
    if torch.all(bp_margin == 0):
        return result.sum(dim=1).sum(dim=1), scale_factors

    return result.sum(dim=1), scale_factors
