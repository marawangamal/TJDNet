from typing import List, Optional
from tjdnet.types import PositivityFuncType
import torch


def sample_topk(p: torch.Tensor, top_k: int, num_samples: int = 1) -> torch.Tensor:
    """Sample from the top-k tokens in distribution `p`.

    Args:
        p (torch.Tensor): Probabilities of shape (B, V).
        top_k (int): Number of top-k tokens to sample from.

    Returns:
        torch.Tensor: Sampled tokens of shape (B, num_samples).
    """
    batch_size = p.size(0)
    top_k_scores, top_k_indices = torch.topk(
        p, k=min(top_k, p.size(1)), dim=1
    )  # (B, top_k)
    top_k_probs = torch.softmax(top_k_scores, dim=1)  # (B, top_k)
    sampled_indices = torch.stack(
        [
            torch.multinomial(top_k_probs[b], num_samples=num_samples)
            for b in range(batch_size)
        ]
    )  # (B, 1)
    next_token = torch.gather(
        top_k_indices, dim=1, index=sampled_indices
    )  # (B, num_samples)
    return next_token


# ----------------------------- runner ------------------------------------
def pad_seqs(
    seqs: List[torch.Tensor],
    pad_token: int,
    pad_length: Optional[int] = None,
):
    """Pad sequences to the same length.

    Args:
        seqs (List[torch.Tensor]): List of sequences of shape (L, *).
        pad_token (int): Padding token.
        pad_length (Optional[int], optional): Length to pad to. Defaults to max length.

    Returns:
        torch.Tensor: Padded sequences of shape (B, L_max, *). (L_max = pad_length or max length)
    """
    pad_length = pad_length if pad_length is not None else max(len(seq) for seq in seqs)
    dummy_seq = torch.zeros(
        pad_length, *seqs[0].shape[1:], dtype=seqs[0].dtype, device=seqs[0].device
    )
    seqs_padded = torch.nn.utils.rnn.pad_sequence(
        (seqs + [dummy_seq]),
        batch_first=True,
        padding_value=pad_token,
    )
    return seqs_padded[:-1]


def get_positional_encodings(
    seq_len: int,
    d_model: int,
    device: torch.device,
):

    # same size with input matrix (for adding with input matrix)
    encoding = torch.zeros(seq_len, d_model, device=device, requires_grad=False)
    pos = torch.arange(0, seq_len, device=device)
    pos = pos.float().unsqueeze(dim=1)
    # 1D => 2D unsqueeze to represent word's position

    _2i = torch.arange(0, d_model, step=2, device=device).float()
    # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
    # "step=2" means 'i' multiplied with two (same with 2 * i)

    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    end_pos = encoding[:, 1::2].size(1)
    encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))[:, :end_pos]

    return encoding  # (seq_len, d_model)


# NOTE:
# Materialized tensor shapes:
# - alpha: (B, R)
# - beta: (B, R)
# - core: (B, H, R, D, R)
# - res_left: (B, R)
# - res_right: (B, R)
# - res_free: (B, R, D, R)


def diagnose(tens: torch.Tensor, tens_name: str = "tensor"):
    assert not torch.isnan(
        tens
    ).any(), f"NaN found in {tens_name} -- (min: {tens.min()}, max: {tens.max()})"

    assert not torch.isinf(
        tens
    ).any(), f"Inf found in {tens_name} -- (min: {tens.min()}, max: {tens.max()})"


def mem_check(msg: str = "unknown"):
    print(f"MEM [{msg}]: {torch.cuda.memory_allocated()/1e9:.2f} GB")
