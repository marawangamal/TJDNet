from typing import Callable, List, Tuple

import torch


# def get_candidates(
#     seqs: List[List],
#     log_probs: torch.Tensor,
#     next_token_log_probs: torch.Tensor,
#     num_beams: int,
# ) -> List[Tuple[list, float]]:
#     """
#     Helper function to extend sequences in beam search.
#     Handles empty sequences at start of generation.
#     """
#     vocab_size = next_token_log_probs.size(1)

#     # Calculate combined scores
#     candidate_scores = log_probs.unsqueeze(1) + next_token_log_probs

#     # Get top candidates
#     flat_scores = candidate_scores.view(-1)
#     top_scores, top_indices = torch.topk(
#         flat_scores, k=min(num_beams, len(flat_scores))
#     )

#     # Convert indices
#     prev_seq_idx = top_indices // vocab_size
#     token_idx = top_indices % vocab_size

#     # Build new candidates - handle empty seqs case
#     new_candidates = []
#     for i in range(len(top_scores)):
#         # For empty seqs, just use the new token
#         # Otherwise extend the previous sequence
#         if len(seqs) == 1 and not seqs[0]:
#             new_sequence = [token_idx[i].item()]
#         else:
#             prev_sequence = list(seqs[prev_seq_idx[i]])
#             new_sequence = prev_sequence + [token_idx[i].item()]

#         new_score = top_scores[i].item()
#         new_candidates.append((new_sequence, new_score))

#     return new_candidates


def get_candidates(
    seqs: List[List],
    log_probs: torch.Tensor,
    next_token_log_probs: torch.Tensor,
    num_beams: int,
) -> List[Tuple[list, float]]:
    """
    Helper function to extend sequences in beam search.

    Args:
        seqs: Current sequences in beam (n_beams, sequence_length)
        log_probs: Current sequence scores (n_beams,)
        next_token_scores: Scores for next tokens (n_beams, vocab_size)
        num_beams: Number of sequences to keep

    Returns:
        List of (new_sequence, new_score) tuples
    """
    # Calculate combined scores
    candidate_scores = (
        log_probs.unsqueeze(1) + next_token_log_probs
    )  # (n_beams, vocab_size)
    vocab_size = next_token_log_probs.size(1)

    # Get top candidates
    flat_scores = candidate_scores.view(-1)
    top_scores, top_indices = torch.topk(
        flat_scores, k=min(num_beams, len(flat_scores))
    )

    # Convert flat indices to sequence and token indices
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
    expand_fn: Callable,  # Function that takes List[Tuple[seq, score]] and returns List[Tuple[seq, score]]
    initial_beam: List[Tuple[List, float]],
    num_beams: int,
    max_steps: int,
) -> Tuple[list, float]:
    """
    Simple beam search that works with any sequence type.

    Args:
        expand_fn: Function that takes current beam and returns list of candidates
        initial_beam: Starting [(sequence, score)] list
        num_beams: Beam width
        max_steps: Maximum steps
    """
    beam = initial_beam

    for _ in range(max_steps):
        if not beam:
            break

        # Get all candidates from current beam
        candidates = expand_fn(beam)

        # Keep top candidates
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:num_beams]

    return beam[0] if beam else ([], float("-inf"))
