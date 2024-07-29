import torch
from typing import Dict, Optional

from TJDNet.TJDLayer.TTDist import TTDist


def get_experiment_name(
    configs: Dict,
    abbrevs: Optional[Dict] = None,
) -> str:
    """Create an experiment name from the configuration dictionary.

    Args:
        configs (Dict): Experiment configuration dictionary.
        abbrevs (Optional[Dict], optional): Working dictionary of abbreviations used in recursive calls. Defaults to None.
        mode (str, optional): Return mode. Defaults to "dict".

    Raises:
        ValueError: Abbreviation not found for key.

    Returns:
        str: Experiment name.
    """

    if abbrevs is None:
        abbrevs = {}

    for key, value in configs.items():
        if isinstance(value, dict):
            get_experiment_name(value, abbrevs=abbrevs)
        else:
            i = 1
            while i <= len(key):
                if key[:i] not in abbrevs:
                    abbrevs[key[:i]] = (
                        str(value)
                        .replace(" ", "")
                        .replace(",", "_")
                        .replace("[", "")
                        .replace("]", "")
                    )
                    break
                i += 1

                if i == len(key) + 1:
                    raise ValueError(
                        "Could not find a suitable abbreviation for key: {}".format(key)
                    )

    return "_".join(["{}{}".format(k, v) for k, v in abbrevs.items()])


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
