from typing import Callable, List, Optional, Tuple

import torch
import line_profiler


# TODO: scores are log probs, does this cause any issues with softmax?
def get_candidates(
    seqs: List[List],
    seq_log_probs: torch.Tensor,
    next_token_probs: torch.Tensor,
    num_beams: int,
    do_sample: bool = True,
    top_k: int = 50,
) -> List[Tuple[list, float]]:
    """
    Helper function to extend sequences with top-k sampling.

    Args:
        seqs: Current sequences in beam (n_beams, sequence_length)
        seq_log_probs: Current sequence scores (n_beams,)
        next_token_probs: Log probabilities for next tokens (n_beams, vocab_size)
        num_beams: Number of sequences to keep
        do_sample: Whether to use sampling (True) or deterministic selection (False)
        top_k: Number of top tokens to consider for sampling
    """
    # Calculate combined scores
    candidate_scores = seq_log_probs.unsqueeze(1) + torch.log(
        next_token_probs
    )  # (n_beams, vocab_size)
    vocab_size = next_token_probs.size(1)

    if do_sample:
        # Get top-k tokens for each sequence
        flat_scores = candidate_scores.view(-1)
        top_scores, top_indices = torch.topk(
            flat_scores, k=min(top_k, vocab_size)
        )  # (n_beams*top_k,)

        # Convert to probabilities and sample
        probs = torch.softmax(top_scores, dim=0)  # (n_beams*top_k)
        sampled_indices = torch.multinomial(probs, num_samples=num_beams)  # (n_beams,)

        # Get selected tokens and their scores
        top_scores = torch.index_select(top_scores, 0, sampled_indices)
        top_indices = torch.index_select(top_indices, 0, sampled_indices)

        # Convert flat indices back to beam and token indices
        prev_seq_idx = top_indices // vocab_size
        token_idx = top_indices % vocab_size

    else:
        # Original beam search logic
        flat_scores = candidate_scores.view(-1)
        top_scores, top_indices = torch.topk(
            flat_scores, k=min(num_beams, len(flat_scores))
        )
        prev_seq_idx = top_indices // vocab_size
        token_idx = top_indices % vocab_size

    # Build new candidates
    new_candidates = []
    for i in range(len(top_scores)):
        prev_sequence = list(seqs[prev_seq_idx[i]])
        new_token = token_idx[i].item()
        new_sequence = prev_sequence + [new_token]
        new_score = top_scores[i].item()
        new_candidates.append((new_sequence, new_score))

    return new_candidates


def beam_search(
    expand_fn: Callable,
    initial_beam: List[Tuple[List, float]],
    num_beams: int,
    max_steps: int,
    stop_token: Optional[int] = None,
) -> Tuple[list, float]:
    active_beams = initial_beam
    finished_beams = []

    for _ in range(max_steps):
        if not active_beams:
            break

        candidates = expand_fn(active_beams)

        # Split candidates into finished and active
        new_active = []
        for seq, score in candidates:
            if stop_token is not None and seq and seq[-1] == stop_token:
                finished_beams.append((seq, score))
            else:
                new_active.append((seq, score))

        # Keep top active beams
        active_beams = sorted(new_active, key=lambda x: x[1], reverse=True)[:num_beams]

    # Return best sequence from finished or active beams
    all_beams = finished_beams + active_beams
    return max(all_beams, key=lambda x: x[1]) if all_beams else ([], float("-inf"))
