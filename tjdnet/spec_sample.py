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
    """Batched speculative sampling for faster text generation.

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


def speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    is_done_candidate,
):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if is_done_candidate and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat(
                (new_candidate_input_ids[:, :n_matches], t), dim=-1
            )
        else:
            valid_tokens = t

    return valid_tokens, n_matches
