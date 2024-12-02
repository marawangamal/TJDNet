import torch

from tensorops.common import get_breakpoints


def select_from_mps_tensor(
    alpha: torch.Tensor, beta: torch.Tensor, core: torch.Tensor, indices: torch.Tensor
):
    """Selects element from a MPS tensor representation (batched).

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (B, R)
        beta (torch.Tensor): Beta tensor of shape (B, R)
        core (torch.Tensor): Core tensor of shape (B, H, R, D, R)
        indices (torch.Tensor): Indices to select from the tensor of shape (B, H). `H` is horizon

    Returns:
        torch.Tensor: Selected elements of shape (B,)
    """
    batch_size, horizon, rank_size, vocab_size, _ = core.shape
    result = alpha
    scale_factors = []
    for t in range(horizon):
        core_reshape = (
            core[:, t]
            .permute(
                0,
                1,
                3,
                2,
            )
            .reshape(-1, vocab_size)
        )
        indices_repeated = (
            indices[:, t]
            .reshape(-1, 1, 1, 1)
            .repeat(1, rank_size, rank_size, 1)
            .reshape(-1, 1)
        )
        core_select = torch.gather(
            core_reshape, 1, indices_repeated
        )  # (BRR, D) -> (BRR, 1)
        core_select = core_select.contiguous()
        # OLD:
        result_raw = torch.einsum(
            "bi, bij -> bj", result, core_select.view(batch_size, rank_size, rank_size)
        )
        scale_factor = torch.linalg.norm(result_raw, dim=-1)  # (B,)
        scale_factors.append(scale_factor)
        result = result_raw / scale_factor.unsqueeze(1)

        # NEW:
        # result = torch.bmm(
        #     result.reshape(batch_size, 1, rank_size),
        #     core_select.view(batch_size, rank_size, rank_size),
        # )  # (B, 1, R) x (B, R, R) -> (B, 1, R)

    # OLD:
    result = torch.einsum("bi, bi -> b", result, beta)

    # NEW:
    # result = torch.bmm(
    #     result.reshape(batch_size, 1, rank_size),
    #     beta.reshape(batch_size, rank_size, 1),
    # ).reshape(
    #     -1
    # )  # (B, 1, R) x (B, R, 1) -> (B, 1, 1) -> (B,)
    return result, scale_factors


def sum_mps_tensor(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
):
    """Sum all elements of a uMPS tensor representation (batched).

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (B, R)
        beta (torch.Tensor): Beta tensor of shape (B R)
        core (torch.Tensor): Core tensor of shape (B, H, R, D, R)
        indices (torch.Tensor): Indices to select from the tensor of shape (B, H). `H` is horizon

    Returns:
        torch.Tensor: Selected elements of shape (B,)
    """
    batch_size, horizon, rank_size, vocab_size, _ = core.shape
    core_margin = core.sum(dim=3)  # (B, H, R, R)
    result = alpha
    scale_factors = []
    for t in range(horizon):
        # OLD:
        result_raw = torch.einsum("bi, bij -> bj", result, core_margin[:, t])
        scale_factor = torch.linalg.norm(result_raw, dim=-1)
        scale_factors.append(scale_factor)
        result = result_raw / scale_factor.unsqueeze(1)

        # NEW:
        # result = torch.bmm(
        #     result.reshape(batch_size, 1, rank_size),
        #     core_margin[:, t],
        # )

    # OLD:
    result = torch.einsum("bi, bi -> b", result, beta)

    # NEW:
    # result = torch.bmm(
    #     result.reshape(batch_size, 1, rank_size),
    #     beta.reshape(batch_size, rank_size, 1),
    # ).reshape(-1)
    return result, scale_factors


def sample_from_mps_tensor(
    alpha: torch.Tensor, beta: torch.Tensor, core: torch.Tensor
) -> torch.Tensor:
    """Samples from an MPS tensor representation of probabilities.

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (R)
        beta (torch.Tensor): Beta tensor of shape (R)
        core (torch.Tensor): Core tensor of shape (H, R, D, R)

    Returns:
        torch.Tensor: Sampled tensor of shape (H,)
    """
    selected_indices = []
    horizon, _, _, _ = core.shape
    for t in range(horizon):
        # Unnormalized P(y_t | y_{<t})
        p_tilde_yt_given_prev, _ = select_margin_mps_tensor(
            alpha=alpha,
            beta=beta,
            core=core,
            ops=torch.tensor(
                selected_indices + [-1] + [-2] * (horizon - t - 1),
                device=alpha.device,
            ),
        )  # (D,)
        # Sample from P(y_t | y_{<t})
        selected_index = torch.multinomial(p_tilde_yt_given_prev, num_samples=1).item()
        selected_indices.append(selected_index)
    return torch.tensor(selected_indices, device=alpha.device)


def select_margin_mps_tensor(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    ops: torch.Tensor,
):
    """Performs selection and marginalization operations on a MPS tensor representation.

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (R)
        beta (torch.Tensor): Beta tensor of shape (R)
        core (torch.Tensor): Core tensor of shape (H, R, D, R)
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
    assert len(core.shape) == 4, "Core tensor must be 4D (non-batched)"
    assert len(ops.shape) == 1, "Ops tensor must be 1D (non-batched)"
    assert (ops >= -2).all() and (ops < core.size(0)).all(), "Invalid ops tensor"

    # Note ops must be in the order of select, free, marginalize
    bp_free, bp_margin = get_breakpoints(
        ops.reshape(1, -1)
    )  # (1,), (1,) selects index at which selects end
    bp_free, bp_margin = int(bp_free.item()), int(bp_margin.item())
    assert bp_free < bp_margin, "Invalid ops tensor (select/marginalize order)"

    # 1. Reduce via selection
    horizon, rank_size, vocab_size, _ = core.shape
    result_select = None
    if bp_free > 0:
        result_select = torch.gather(
            core[:bp_free].permute(0, 1, 3, 2).reshape(-1, vocab_size),
            dim=1,
            index=ops[:bp_free]
            .reshape(-1, 1, 1)
            .repeat(1, rank_size, rank_size)
            .reshape(-1, 1),
        )
        result_select = torch.linalg.multi_dot(
            [t for t in result_select]
        )  # (R, R) x H' => (R, R

    # 2. Reduce via marginalization
    result_margin = None
    if bp_margin < horizon:
        result_margin = core[bp_margin:].sum(dim=2)  # (H'', R, R)
        result_margin = torch.linalg.multi_dot(
            [t for t in result_margin]
        )  # (R, R) x H'' => (R, R)

    # 3. Combine results
    return (
        mps_to_tensor(
            alpha=alpha @ result_select if result_select is not None else alpha,
            beta=result_margin @ beta if result_margin is not None else beta,
            core=core[bp_free:bp_margin],
        ),
        [],
    )


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
    """
    full_shape = [core.size(2)] * core.size(0)
    result = torch.einsum("r, rdj -> dj", alpha, core[0])
    for t in range(1, core.size(0)):
        result = torch.einsum("kr, rdj -> kdj", result, core[t])
        result = result.reshape(-1, core.size(1))
    result = torch.einsum("kr,r -> k", result, beta)
    return result.reshape(full_shape)


def materialize_mps_tensor(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    is_normalized: bool = False,
):
    """Materialize a MPS tensor network (batched version).

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (B, R)
        beta (torch.Tensor): Beta tensor of shape (B, R)
        core (torch.Tensor): Core tensor of shape (B, H, R, D, R)

    Returns:
        torch.Tensor: Materialized tensor of shape (B, D, D ... D) with `n_core_repititions` dimensions
    """
    batch_size, horizon, rank_size, vocab_size, _ = core.shape

    result = torch.einsum(
        "bi, bidj->bdj",
        alpha,
        core[:, 0],
    )

    for t in range(1, horizon):
        result = torch.einsum(
            "bdi, bivj->bdvj",
            result,
            core[:, t],
        )
        result = result.reshape(batch_size, -1, rank_size)

    result = torch.einsum(
        "bdj, bj->bd",
        result,
        beta,
    )

    # Break out all vocab_size dimensions
    result = result.reshape(batch_size, *[vocab_size for _ in range(horizon)])

    if is_normalized:
        norm_const = (
            result.reshape(batch_size, -1)
            .sum(1)
            .reshape(tuple([batch_size, *[1 for _ in range(horizon)]]))
        )
        result = result / norm_const

    return result
