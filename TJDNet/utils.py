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


def umps_batch_select_marginalize(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    n_core_repititions: int,
    selection_ids: Dict[int, int],
    marginalize_ids: List[int],
):
    """Given a uMPS, perform select and/or marginalize operations.

    Args:
        alpha (torch.Tensor): Parameter tensor. Shape: (B, R).
        beta (torch.Tensor): Parameter tensor. Shape: (B, R).
        core (torch.Tensor): Core tensor. Shape: (R, D, R).
        n_core_repititions (int): Number of core repetitions.
        selection_ids (List[int]): Shape: (B, s1).
        marginalize_ids (List[int]):Shape: (B, s2).

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
