import torch
import line_profiler


# TODO: Try with gather
# TODO: try this -> core_select.contiguous()
# TODO: try torch.bmm
# Make contiguous
def select_from_umps_tensor(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    indices: torch.Tensor,
):
    """Selects element from a uMPS tensor representation (batched).

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (B, R)
        beta (torch.Tensor): Beta tensor of shape (B R)
        core (torch.Tensor): Core tensor of shape (B, R, D, R)
        indices (torch.Tensor): Indices to select from the tensor of shape (B, H). `H` is horizon

    Returns:
        torch.Tensor: Selected elements of shape (B,)
    """
    batch_size, rank_size, vocab_size, _ = core.shape
    result = alpha
    scale_factors = []
    for t in range(indices.shape[1]):
        core_reshape = core.permute(
            0,
            1,
            3,
            2,
        ).reshape(-1, vocab_size)
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
        result_raw = torch.einsum(
            "bi, bij -> bj", result, core_select.view(batch_size, rank_size, rank_size)
        )
        scale_factor = torch.linalg.norm(result_raw, dim=-1)  # (B,)
        scale_factors.append(scale_factor)
        result = result_raw / scale_factor.unsqueeze(1)
    result = torch.einsum("bi, bi -> b", result, beta)
    return result, scale_factors


# @line_profiler.profile
def sum_umps_tensorV2(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    n_core_repititions: int,
):
    """Sum all elements of a uMPS tensor representation (batched).

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (B, R)
        beta (torch.Tensor): Beta tensor of shape (B R)
        core (torch.Tensor): Core tensor of shape (B, R, D, R)
        indices (torch.Tensor): Indices to select from the tensor of shape (B, H). `H` is horizon

    Returns:
        torch.Tensor: Selected elements of shape (B,)
    """
    batch_size = alpha.size(0)
    core_margin = core.sum(dim=2)  # (B, R, R)
    result = alpha
    scale_factors = []
    for t in range(n_core_repititions):
        result_raw = torch.einsum("bi, bij -> bj", result, core_margin)
        scale_factor = torch.linalg.norm(result_raw, dim=-1)
        scale_factors.append(scale_factor)
        result = result_raw / scale_factor.unsqueeze(1)
    result = torch.einsum("bi, bi -> b", result, beta)
    return result, scale_factors


def sum_umps_tensor(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    operation_map: torch.Tensor,
    apply_scale_factors: bool = False,
    reversed: bool = False,
    skip_last: bool = False,
):
    """Given a uMPS, perform select and/or marginalize operations (batched version).

    Args:
        alpha (torch.Tensor): Parameter tensor. Shape: (B, R).
        beta (torch.Tensor): Parameter tensor. Shape: (B, R).
        core (torch.Tensor): Core tensor. Shape: (B, R, D, R).
        operation_map (torch.Tensor): Operations to perform on indices. Shape: (B, N).  -1 for marginalize, [0, V) for select

    Returns:
        torch.Tensor: Evaluation of the uMPS tensor network. Shape: (B,)

    """
    # Input validation: alpha, beta, core
    assert len(alpha.shape) == 2, "Alpha should be a 2D tensor"
    assert len(beta.shape) == 2, "Beta should be a 2D tensor"
    assert len(core.shape) == 4, "Core should be a 4D tensor"

    # Input validation: selection_map, marginalize_mask
    assert len(operation_map.shape) == 2, "Operation map should be a 2D tensor"

    free_legs = (operation_map == -2).sum(dim=-1)
    assert torch.all(free_legs.sum(dim=-1) <= 1), "Must have at most one free leg"
    core_margins = core.sum(dim=2)

    n_core_repititions = operation_map.shape[1]
    res_tmp = beta if reversed else alpha
    scale_factors = []

    def get_contracted_core(
        core: torch.Tensor, batch_idx: int, time_idx: int, operation_map: torch.Tensor
    ):
        if operation_map[batch_idx, time_idx] >= 0:  # Select
            return core[batch_idx, :, operation_map[batch_idx, time_idx], :]
        elif operation_map[batch_idx, time_idx] == -1:  # Marginalize
            return core_margins[batch_idx]
        else:  # Not accepted
            raise ValueError("Invalid operation")

    for t in range(n_core_repititions):
        res_tmp_prime = torch.stack(
            [
                get_contracted_core(core, b, t, operation_map)
                for b in range(core.shape[0])
            ]
        )  # (B, R, R)
        z_tmp = torch.linalg.norm(res_tmp, dim=1)
        scale_factors.append(z_tmp)
        res_tmp = res_tmp / z_tmp.unsqueeze(1)
        if reversed:
            res_tmp = torch.einsum("bij, bi->bj", res_tmp_prime, res_tmp)
        else:
            res_tmp = torch.einsum("bi,bij->bj", res_tmp, res_tmp_prime)
    if skip_last:
        return res_tmp, scale_factors
    res = torch.einsum("bi, bi -> b", res_tmp, alpha if reversed else beta)
    if apply_scale_factors:
        norm_const = torch.stack(scale_factors, dim=1).prod(dim=1)
        res = res * norm_const
        scale_factors = []
    return res, scale_factors


def materialize_umps_tensor(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    n_core_repititions: int,
    is_normalized: bool = False,
):
    """Materialize a uMPS tensor network (batched version).

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (B, R)
        beta (torch.Tensor): Beta tensor of shape (B, R)
        core (torch.Tensor): Core tensor of shape (B, R, D, R)

    Raises:
        NotImplementedError: _description_

    Returns:
        torch.Tensor: Materialized tensor of shape (B, D, D ... D) with `n_core_repititions` dimensions
    """
    batch_size, rank_size, vocab_size, _ = core.shape

    result = torch.einsum(
        "bi, bidj->bdj",
        alpha,
        core,
    )

    for i in range(1, n_core_repititions):
        result = torch.einsum(
            "bdi, bivj->bdvj",
            result,
            core,
        )
        result = result.reshape(batch_size, -1, rank_size)

    result = torch.einsum(
        "bdj, bj->bd",
        result,
        beta,
    )

    # Break out all vocab_size dimensions
    result = result.reshape(
        batch_size, *[vocab_size for _ in range(n_core_repititions)]
    )

    if is_normalized:
        norm_const = (
            result.reshape(batch_size, -1)
            .sum(1)
            .reshape(tuple([batch_size, *[1 for _ in range(n_core_repititions)]]))
        )
        result = result / norm_const

    return result
