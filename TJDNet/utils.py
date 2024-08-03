from typing import Optional, Dict, List
import torch

from TJDNet import TTDist


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
    if torch.isnan(x).any() or torch.isinf(x).any():
        if raise_error:
            raise ValueError(f"NaN/Inf detected in tensor: {msg}")
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


def get_init_params_uniform_std_positive(batch_size, rank, output_size, vocab_size):
    # TODO: Add noise to the identities
    # TODO: Pass alpha  and beta through renormalization layer
    # TODO: Parameterize the residuals
    alpha = (
        torch.randn(1, rank).repeat(batch_size, 1) * torch.sqrt(torch.tensor(1 / rank))
    ).abs()
    beta = (
        torch.randn(1, rank).repeat(batch_size, 1) * torch.sqrt(torch.tensor(1 / rank))
    ).abs()
    core = torch.nn.Parameter(
        torch.eye(rank)
        .unsqueeze(1)
        .repeat(1, vocab_size, 1)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


def get_init_params_onehot(
    batch_size: int, rank: int, vocab_size: int, onehot_idx: int = 1, *args, **kwargs
):
    """Create initial parameters for a TT distribution with a one-hot cores.

    Args:
        batch_size (int): Batch size.
        rank (int): Rank of the TT decomposition.
        vocab_size (int): Vocabulary size.
        onehot_idx (int): Index of the one-hot core.

    Returns:
        _type_: _description_
    """
    alpha = torch.ones(1, rank).repeat(batch_size, 1)
    beta = torch.ones(1, rank).repeat(batch_size, 1)
    coreZero = torch.zeros(rank, vocab_size, rank)
    coreOneHot = torch.zeros(rank, vocab_size, rank)
    coreOneHot[:, onehot_idx, :] = torch.eye(rank)
    core = torch.nn.Parameter(
        (coreZero + coreOneHot).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


def get_init_params_randn_positive(batch_size, rank, vocab_size, *args, **kwargs):
    alpha = (torch.randn(1, rank).repeat(batch_size, 1)).abs()
    beta = (torch.randn(1, rank).repeat(batch_size, 1)).abs()
    core = torch.nn.Parameter(
        torch.randn(rank, vocab_size, rank)
        .abs()
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


def get_preference_loss(
    ttdist: TTDist,
    samples: torch.Tensor,
    eps: float = 1e-6,
    vocab_size: int = 4,
    neg_samples_multiplier: int = 1000,
    num_neg_batches: int = 10,
):
    """_summary_

    Args:
        ttdist (TTDist): Instance of TTDist.
        samples (torch.Tensor): Samples over which to maximize the preference. Shape: (batch_size, seq_len).
        eps (float, optional): _description_. Defaults to 1e-6.
        vocab_size (int, optional): _description_. Defaults to 4.
        neg_samples_multiplier (int, optional): _description_. Defaults to 1000.
        num_neg_batches (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    samples_pos = samples
    batch_size = samples_pos.shape[0]
    seq_len = samples_pos.shape[1]
    # make a batch of negative samples by creating rand tensor of size batch_size x seq_len with indices in [0, vocab_size-1]
    probs_tilde_pos = ttdist.get_prob(samples_pos)
    # probs_tilde_neg, norm_constant_neg = ttdist.get_prob_and_norm(samples_neg)
    probs_tilde_neg_lst = [
        ttdist.get_prob(
            torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len),
                dtype=torch.long,
                device=samples.device,
            )
        )
        for _ in range(num_neg_batches)
    ]

    probs_tilde_neg_sum = torch.stack(probs_tilde_neg_lst, dim=0).sum(dim=0)
    preference_loss = -torch.log(probs_tilde_pos + eps) + torch.log(
        probs_tilde_neg_sum + eps
    )
    preference_loss = preference_loss.mean()

    return preference_loss, probs_tilde_neg_sum


def get_entropy_loss(
    ttdist: TTDist, samples: torch.Tensor, eps: float = 1e-6, vocab_size: int = 4
):
    probs_tilde, norm_constant = ttdist.get_prob_and_norm(samples)
    # retain_grad on probs_tilde to compute the gradient of the loss w.r.t. probs_tilde
    probs_tilde.retain_grad()
    norm_constant.retain_grad()
    loss = (-torch.log(probs_tilde + eps) + torch.log(norm_constant)).mean()
    return loss, norm_constant
