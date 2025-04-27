from typing import Callable, List, Optional, Tuple
import torch

from tjdnet.distributions._base import BaseDistribution


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


def spec_sample(
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    sample_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    """Batched specualtive sampling.

    Args:
        p_target (torch.Tensor): Target probabilities of shape (B, V).
        p_draft (torch.Tensor): Draft probabilities of shape (B, V).
        sample_fn (Optional[Callable[[torch.Tensor], torch.Tensor]], optional): Function to sample from batch of probabilities.
            If None, greedy sampling is used. Defaults to None.
    """
    sample_fn = sample_fn or (lambda x: torch.argmax(x, dim=-1))
    y_hat = sample_fn(p_target)  # (B,)

    should_reject = torch.zeros_like(y_hat, dtype=torch.bool)
    maybe_reject = p_draft > p_target
    should_reject[maybe_reject] = torch.bernoulli(
        1 - p_target[maybe_reject] / p_draft[maybe_reject]
    ).bool()
    p_adj = torch.maximum(
        torch.zeros_like(p_target), p_target - p_draft
    ) / torch.linalg.norm(p_target - p_draft, dim=-1)

    y_hat[should_reject] = sample_fn(p_adj[should_reject])
    return y_hat, should_reject


def spec_sample_v2(
    model_p: Callable[[torch.Tensor], torch.Tensor],
    model_q: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    sample_fn: Callable[[torch.Tensor], torch.Tensor],
):
    """Batched speculative sampling for faster text generation.

    Uses draft model guesses q_hat and dist q(y|x) to sample from the target model p(y|x).

    Args:
        model_p (Callable): Target model. Signature: y -> p(y|x)
        model_q (Callable): Draft model. Signature: {} -> y_hat, q(y|x).
        sample_fn (Callable): Sampling function. Takes probabilities (shape B, V),
            returns chosen token index (shape B,).

    Note:
        Key variables and typical shapes:
        (B = Batch size, H = Draft length, V = Vocabulary size)
        - y (torch.Tensor): Predicted token sequence. Shape (B, H).
        - y_hat (torch.Tensor): Draft tokens from model_q. Shape: (B, H).
        - q(y|x) (torch.Tensor): Draft probabilities from model_q. Shape: (B, H, V).
        - p(y|x) (torch.Tensor): Target probabilities from model_p. Shape: (B, H, V).

    Returns:
        torch.Tensor: Accepted tokens. Shape: (B, H'), where H' is the number of
            accepted tokens per sequence (can be > 1 and vary).
    """
    # Get draft preds and probs
    q_hat, qy = model_q()  # (B, H), (B, H, V)
    py = model_p(q_hat)  # (B, H, V)

    batch_size, horizon, vocab_size = py.size()

    # Assert: valid shapes
    assert q_hat.size() == (
        batch_size,
        horizon,
    ), f"y_hat shape mismatch expected {(batch_size, horizon)}, got {q_hat.size()}"
    assert py.size() == (
        batch_size,
        horizon,
        vocab_size,
    ), f"py shape mismatch expected {(batch_size, horizon)}, got {py.size()}"
    # Assert: positive probs
    assert (py >= 0).all(), f"Negative probs in py: {py.min()}"
    assert (qy >= 0).all(), f"Negative probs in qy: {qy.min()}"
    # Assert: valid conditional probs
    assert torch.isclose(
        py.sum(dim=-1), torch.ones((batch_size, horizon), device=py.device)
    ).all(), f"Invalid probs in py: {py.sum(-1)}"

    # Reduce prob tensors. Shape: (B, H, V) -> (B,)
    qy_select = torch.gather(qy, dim=-1, index=q_hat.unsqueeze(-1)).squeeze(-1).prod(-1)
    py_select = torch.gather(py, dim=-1, index=q_hat.unsqueeze(-1)).squeeze(-1).prod(-1)
    y_out = []

    h = 0
    while h < horizon:
        r = torch.rand((batch_size,), device=q_hat.device)
        accept_mask = r < torch.minimum(
            torch.ones_like(py_select, device=py_select.device), (py_select / qy_select)
        )  # (B,)
        if accept_mask.all():
            # all samples accepted
            y_out.append(q_hat[:, h : h + 1])  # (B, 1)
            h += 1
        else:
            # some samples rejected
            y_adj = sample_fn(py[:, h] - qy[:, h])  # (B, V) => (B,)
            y_out.append(y_adj.unsqueeze(1))  # (B, 1)
            break

    return torch.cat(y_out, dim=1)  # (B, H)


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
