from typing import Optional, Dict, List
import torch


def batched_index_select(
    input: torch.Tensor,
    batched_index: torch.Tensor,
):
    """Perform a batched index select operation.

    Args:
        input (torch.Tensor): Input tensor. Shape: (d1, d2, ..., dN).
        batched_index (torch.Tensor): Batched index tensor. Shape: (batch_size, N).

    Returns:
        torch.Tensor: Output tensor. Shape: (batch_size, d2, ..., dN).
    """
    cols = [batched_index[:, i] for i in range(batched_index.shape[1])]
    return input[cols]


def check_naninf(x: torch.Tensor, msg: Optional[str] = None, raise_error: bool = True):
    if raise_error:
        if torch.isnan(x).any():
            raise ValueError(f"NaN detected in tensor: {msg}")
        elif torch.isinf(x).any():
            raise ValueError(f"Inf detected in tensor: {msg}")
    else:
        return True


def umps_select_marginalize_old(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    n_core_repititions: int,
    selection_ids: Dict[int, int],
    marginalize_ids: List[int],
):
    """Given a uMPS, perform select and/or marginalize operations.

    Args:
        alpha (torch.Tensor): Parameter tensor. Shape: (R).
        beta (torch.Tensor): Parameter tensor. Shape: (R).
        core (torch.Tensor): Core tensor. Shape: (R, D, R).
        n_core_repititions (int): Number of core repetitions.
        selection_ids (Dict[int, int]): Dictionary of selection indices.
        marginalize_ids (List[int]) : List of marginalization indices.

    Returns:
         torch.Tensor: Evaluation of the uMPS tensor network. Shape: (B, n_core_repititions - (s1 + s2)).

    """

    # Validation
    assert len(alpha.shape) == 1, "Alpha should be a 1D tensor"
    assert len(beta.shape) == 1, "Beta should be a 1D tensor"

    # Can't have same index in both selection and marginalization
    assert not any(
        [sid in marginalize_ids for sid in selection_ids.keys()]
    ), "Can't have same index in both selection and marginalization"

    # Can't have indices out of range
    assert all(
        [sid < n_core_repititions for sid in selection_ids.keys()]
    ), "Selection index out of range"

    assert all(
        [mid < n_core_repititions for mid in marginalize_ids]
    ), "Marginalization index out of range"

    result = None
    core_margin = torch.einsum(
        "idj,d->ij", core, torch.ones(core.shape[1], device=core.device)
    )
    for i in range(n_core_repititions):
        if i in selection_ids:
            sid = selection_ids[i]
            node = core[:, sid, :]
        elif i in marginalize_ids:
            node = core_margin
        else:
            node = core

        if result is None:
            result = node
        elif len(node.shape) == 2:
            shape_init = result.shape
            result = result.reshape(-1, shape_init[-1]) @ node
            result = result.reshape(tuple(shape_init[:-1]) + tuple(node.shape[1:]))
        else:
            shape_init = result.shape
            result_tmp = result.reshape(-1, shape_init[-1])
            result = torch.einsum("ij,jdl->idl", result_tmp, node)
            result = result.reshape(tuple(shape_init[:-1]) + tuple(node.shape[1:]))

    # Contract with alpha and beta
    if result is None:
        raise ValueError("No core nodes selected or marginalized")

    shape_init = result.shape
    result = result.reshape(shape_init[0], -1, shape_init[-1])
    result = torch.einsum("i,idj,j->d", alpha, result, beta)
    result = result.reshape(tuple(shape_init[1:-1]))
    return result


def umps_select_marginalize_batched(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    selection_map: torch.Tensor,
    marginalize_mask: torch.Tensor,
    apply_scale_factor: bool = True,
):
    """Given a uMPS, perform select and/or marginalize operations (batched version).

    Args:
        alpha (torch.Tensor): Parameter tensor. Shape: (B, R).
        beta (torch.Tensor): Parameter tensor. Shape: (B, R).
        core (torch.Tensor): Core tensor. Shape: (B, R, D, R).
        selection_map (torch.Tensor): Batched selection indices. Negative denote non-select indices. Shape: (B, N).
        marginalize_mask (torch.Tensor): Batched marginalization mask. Shape: (B, N).

    Returns:
         torch.Tensor: Evaluation of the uMPS tensor network. Shape: (B, n_core_repititions - (s1 + s2)).

    """
    # FIXME: Does not support interleave selection and marginalization
    # Validation
    assert len(alpha.shape) == 2, "Alpha should be a 2D tensor"
    assert len(beta.shape) == 2, "Beta should be a 2D tensor"
    assert len(core.shape) == 4, "Beta should be a 4D tensor"
    assert len(selection_map.shape) == 2, "Selection map should be a 2D tensor"
    assert len(marginalize_mask.shape) == 2, "Marginalize mask should be a 2D tensor"
    assert (
        selection_map.shape == marginalize_mask.shape
    ), "Selection and marginalize mask should have same shape"
    free_legs = torch.logical_and(selection_map == -1, marginalize_mask == 0)
    assert torch.all(free_legs.sum(dim=-1) == 1), "Must have excatly one free leg"

    batch_size, rank, vocab_size, _ = core.shape
    n_core_repititions = selection_map.shape[1]

    res_left = alpha
    res_right = beta

    core_margins = torch.einsum(
        "bijk,bj->bik",
        core,
        torch.ones(batch_size, vocab_size, device=core.device),
    )

    norm_consts = []

    for t in range(n_core_repititions):
        res_left_prime = torch.stack(
            [
                (
                    core[b, :, selection_map[b, t], :]
                    if selection_map[b, t] >= 0
                    else torch.eye(rank, device=core.device)
                )
                for b in range(core.shape[0])
            ]
        )  # (B, R, R)
        # z_tmp = res_left.sum(dim=1) replace with norm of a vec
        z_tmp = torch.linalg.norm(res_left, dim=1)
        norm_consts.append(z_tmp)
        res_left = res_left / z_tmp.unsqueeze(1)
        res_left = torch.einsum("bi,bij->bj", res_left, res_left_prime)

    for t in range(n_core_repititions):
        res_right_prime = torch.stack(
            [
                (
                    core_margins[b]
                    if marginalize_mask[b, t]
                    else torch.eye(rank, device=core.device)
                )
                for b in range(core.shape[0])
            ]
        )
        # z_tmp = res_right.sum(dim=1)
        z_tmp = torch.linalg.norm(res_left, dim=1)
        norm_consts.append(z_tmp)
        res_right = res_right / z_tmp.unsqueeze(1)
        res_right = torch.einsum("bij, bj->bi", res_right_prime, res_right)
    res = torch.einsum("bi, bidj, bj -> bd", res_left, core, res_right)
    if apply_scale_factor:
        norm_const = torch.stack(norm_consts, dim=1).prod(dim=1)
        res = res * norm_const.unsqueeze(1)
    return res, norm_consts


def umps_materialize_batched(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    n_core_repititions: int,
    normalize: bool,
):
    """Materialize the joint distribution of a uMPS.

    Args:
        alpha (torch.Tensor): Parameter tensor. Shape: (B, R).
        beta (torch.Tensor): Parameter tensor. Shape: (B, R).
        core (torch.Tensor): Core tensor. Shape: (B, R, D, R).
        n_core_repititions (int): Number of core repetitions.
        normalize (bool, optional): Normalize the joint distribution. Defaults to True.

    Returns:
        torch.Tensor: Materialized joint distribution. Shape: (B, D, D, ..., D)
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
    if normalize:
        norm_const = (
            result.reshape(batch_size, -1)
            .sum(1)
            .reshape(tuple([batch_size, *[1 for _ in range(n_core_repititions)]]))
        )
        result = result / norm_const
    return result
