from TJDNet import MPSDist, MPSDistBase


import torch


def get_preference_loss(
    ttdist: MPSDist,
    samples: torch.Tensor,
    eps: float = 1e-6,
    vocab_size: int = 4,
    num_neg_batches: int = 10,
    *args,
    **kwargs,
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
    probs_tilde_pos = ttdist.get_unnorm_prob(samples_pos)
    # probs_tilde_neg, norm_constant_neg = ttdist.get_prob_and_norm(samples_neg)
    probs_tilde_neg_lst = [
        ttdist.get_unnorm_prob(
            torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len),
                dtype=torch.long,
                device=samples.device,
            )
        )[0]
        for _ in range(num_neg_batches)
    ]

    probs_tilde_neg_sum = torch.stack(probs_tilde_neg_lst, dim=0).sum(dim=0)
    preference_loss = -torch.log(probs_tilde_pos + eps) + torch.log(
        probs_tilde_neg_sum + eps
    )
    preference_loss = preference_loss.mean()

    return preference_loss


def get_entropy_loss(
    ttdist: MPSDist,
    samples: torch.Tensor,
    eps: float = 1e-6,
    *args,
    **kwargs,
):
    probs_tilde, norm_constant, _, _ = ttdist.get_unnorm_prob_and_norm(samples)
    # retain_grad on probs_tilde to compute the gradient of the loss w.r.t. probs_tilde
    probs_tilde.retain_grad()
    norm_constant.retain_grad()
    loss = (-torch.log(probs_tilde + eps) + torch.log(norm_constant)).mean()
    return loss


def get_entropy_loss_stable(
    ttdist: MPSDistBase,
    targets: torch.Tensor,
    eps: float = 1e-6,
    *args,
    **kwargs,
):
    """Compute entropy loss using MPSDistBase instance. This version is more stable than get_entropy_loss.

    Args:
        ttdist (MPSDistBase): MPSDistBase instance.
        targets (torch.Tensor): Samples over which to compute the entropy loss. Shape: (batch_size, seq_len).
        eps (float, optional): Small value to prevent log(0). Defaults to 1e-6.

    Returns:
        torch.Tensor: Entropy loss.
    """
    probs_tilde, z, p_tilde_scale_factors, z_scale_factors = (
        ttdist.get_unnorm_prob_and_norm(targets, apply_scale_factor=False)
    )
    # Shapes:
    # probs_tilde: (B,) <-- downscaled by elements in z_list_select
    # norm_constant_tilde: (B,) <-- downscaled by elements in z_list_norm
    # z_list: (B, T)
    # z_list_norm: (B, T)
    # Note: normalization constant is correct up to a scale factor
    loss = (
        -torch.log(probs_tilde + eps)
        + torch.log(z)
        - sum([torch.log(z) for z in p_tilde_scale_factors])
        + sum([torch.log(z) for z in z_scale_factors])
    ).mean()
    return loss


def entropy_loss(
    probs_tilde: torch.Tensor,
    targets: torch.Tensor,
    *args,
    **kwargs,
):
    """Compute entropy loss using unnormalized probabilities.

    Args:
        probs_tilde (torch.Tensor): Unnormalized probabilities. Shape: (batch_size, n_classes).
        targets (torch.Tensor): Samples over which to compute the entropy loss. Shape: (batch_size, seq_len).

    Returns:
        torch.Tensor: Entropy loss.
    """
    # (1) No exponentiation (this fails)
    log_z = torch.log(probs_tilde.sum(dim=-1))
    probs_tilde_select = torch.gather(probs_tilde, 1, targets.reshape(-1, 1)).squeeze()
    loss = (-torch.log(probs_tilde_select) + log_z).mean()

    # (2) LogSumExp (this works)
    # log_z = torch.logsumexp(probs_tilde, dim=-1)
    # probs_tilde_select = torch.gather(probs_tilde, 1, targets.reshape(-1, 1)).squeeze()
    # loss = (-probs_tilde_select + log_z).mean()
    return loss
