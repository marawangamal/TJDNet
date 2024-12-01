import torch


def select_from_mps_tensor(
    alpha: torch.Tensor, beta: torch.Tensor, core: torch.Tensor, indices: torch.Tensor
):
    """Selects element from a MPS tensor representation (batched).

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (B, R)
        beta (torch.Tensor): Beta tensor of shape (B R)
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
        result_raw = torch.einsum(
            "bi, bij -> bj", result, core_select.view(batch_size, rank_size, rank_size)
        )
        scale_factor = torch.linalg.norm(result_raw, dim=-1)  # (B,)
        scale_factors.append(scale_factor)
        result = result_raw / scale_factor.unsqueeze(1)
    result = torch.einsum("bi, bi -> b", result, beta)
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
        result_raw = torch.einsum("bi, bij -> bj", result, core_margin[:, t])
        scale_factor = torch.linalg.norm(result_raw, dim=-1)
        scale_factors.append(scale_factor)
        result = result_raw / scale_factor.unsqueeze(1)
    result = torch.einsum("bi, bi -> b", result, beta)
    return result, scale_factors


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
