import torch


from typing import Callable, Tuple


def spec_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalizes the input tensor x to the range [0, 1].

    Args:
        x (torch.Tensor): Input tensor. Shape: (*, V) where * is any number of dimensions.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    z = torch.clamp(x, min=0)
    return z / z.sum(dim=-1, keepdim=True)


def spec_sample(
    model_p: Callable[[torch.Tensor], torch.Tensor],
    model_q: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    sample_fn: Callable[[torch.Tensor], torch.Tensor],
):
    """Batched speculative sampling for faster text generation (https://arxiv.org/pdf/2302.01318).

    Uses draft model guesses q_hat and dist q(y|x) to sample from the target model p(y|x).

    Args:
        model_p (Callable): Target model. Signature: y -> p(y|x)
        model_q (Callable): Draft model. Signature: {} -> y_hat, q(y|x).
        sample_fn (Callable): Sampling function. Mapping: (B, V) -> (B,).

    Note:
        Key variables and typical shapes:
        (B = Batch size, H = Draft length, V = Vocabulary size)
        - y (torch.Tensor): Predicted token sequence. Shape (B, H).
        - y_hat (torch.Tensor): Draft tokens from model_q. Shape: (B, H).
        - q(y|x) (torch.Tensor): Draft probabilities from model_q. Shape: (B, H, V).
        - p(y|x) (torch.Tensor): Target probabilities from model_p. Shape: (B, H, V).

    Returns:
        torch.Tensor: Sampled token sequence. Shape: (B, H') where H" \\in [1, H].
    """
    # Get draft preds and probs
    q_hat, qy = model_q()  # (B, H), (B, H, V)
    py = model_p(q_hat)  # (B, H, V)

    batch_size, horizon, vocab_size = py.size()

    # ============= Input validation ========================================
    checks = [
        # ---- shape checks ----
        {
            "test": lambda: q_hat.shape == (batch_size, horizon),
            "msg": f"y_hat shape mismatch expected {(batch_size, horizon)}, got {q_hat.size()}",
        },
        {
            "test": lambda: py.shape == (batch_size, horizon, vocab_size),
            "msg": f"py shape mismatch expected {(batch_size, horizon)}, got {py.size()}",
        },
        {
            "test": lambda: sample_fn(
                torch.zeros((batch_size, vocab_size), device=py.device)
            ).shape
            == (batch_size,),
            "msg": f"sample_fn output shape mismatch expected {(batch_size,)}",
        },
        # ---- probability checks ----
        {"test": lambda: (py >= 0).all(), "msg": f"Negative probs in py: {py.min()}"},
        {"test": lambda: (qy >= 0).all(), "msg": f"Negative probs in qy: {qy.min()}"},
        {
            "test": lambda: torch.isclose(
                py.sum(dim=-1), torch.ones((batch_size, horizon), device=py.device)
            ).all(),
            "msg": f"invalid probs in py: {py.sum(-1)}",
        },
        # ----- only bs=1 supported currently -----
        {
            "test": lambda: batch_size == 1,
            "msg": f"batch_size > 1 not yet supported: {batch_size}",
        },
    ]
    for check in checks:
        if not check["test"]():
            raise ValueError(check["msg"])
    # =====================================================================

    # Reduce prob tensors. Shape: (B, H, V) -> (B,)
    qy_select = torch.gather(qy, dim=-1, index=q_hat.unsqueeze(-1)).squeeze(-1).prod(-1)
    py_select = torch.gather(py, dim=-1, index=q_hat.unsqueeze(-1)).squeeze(-1).prod(-1)
    y_out = []

    h = 0
    n_accept = 0
    while h < horizon:
        r = torch.rand((batch_size,), device=q_hat.device)
        accept_mask = r < torch.minimum(
            torch.ones_like(py_select, device=py_select.device), (py_select / qy_select)
        )  # (B,)
        if accept_mask.all():
            # all samples accepted
            y_out.append(q_hat[:, h : h + 1])  # (B, 1)
            h += 1
            n_accept += 1
        else:
            # some samples rejected
            # === Debug
            y_adj = sample_fn(py[:, h])
            # ====
            # y_adj = sample_fn(spec_normalize(py[:, h] - qy[:, h]))  # (B, V) => (B,)
            y_out.append(y_adj.unsqueeze(1))  # (B, 1)
            break

    return torch.cat(y_out, dim=1), n_accept
