from typing import List, Optional, Tuple
import torch

from tensorops.common import get_breakpoints, mps_to_tensor
from utils.beam_search import beam_search


def select_from_mps_tensor(
    alpha: torch.Tensor, beta: torch.Tensor, core: torch.Tensor, indices: torch.Tensor
):
    """Selects element from a MPS tensor representation (batched).

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (B, R)
        beta (torch.Tensor): Beta tensor of shape (B, R)
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


def sample_from_mps_tensorV2(
    alpha: torch.Tensor, beta: torch.Tensor, core: torch.Tensor, num_beams: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples from an MPS tensor using beam search.

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (R)
        beta (torch.Tensor): Beta tensor of shape (R)
        core (torch.Tensor): Core tensor of shape (H, R, D, R)
        beam_size (int): Size of beam for search

    Returns:
        torch.Tensor: Most probable sequence found, shape (H,)
    """
    horizon, _, vocab_size, _ = core.shape

    def expand_beam(beam):
        seqs, log_probs = zip(*beam)
        log_probs = torch.tensor(log_probs, device=alpha.device)  # (beam_size,)

        # Get next token probabilities
        all_p_next = []
        for seq in seqs:
            t = len(seq)
            p_next, _ = select_margin_mps_tensor(
                alpha=alpha,
                beta=beta,
                core=core,
                ops=torch.tensor(
                    seq + [-1] + [-2] * (horizon - t - 1), device=alpha.device
                ),
                use_scale_factors=False,
            )
            all_p_next.append(p_next)

        # Calculate scores
        all_p_next = torch.stack(all_p_next)  # (beam_size, vocab_size)
        candidate_scores = log_probs.unsqueeze(1) + torch.log(all_p_next)

        # Get top candidates
        flat_scores = candidate_scores.view(-1)
        top_scores, top_indices = torch.topk(
            flat_scores, k=min(num_beams, len(flat_scores))
        )

        # Build new sequences
        prev_seq_idx = top_indices // vocab_size
        token_idx = top_indices % vocab_size

        # Build new candidates list
        new_candidates = []
        for i in range(len(top_scores)):  # num_beams times
            # Get the previous sequence we're extending
            prev_sequence = list(seqs[prev_seq_idx[i]])
            # Add the new token to it
            new_token = token_idx[i].item()
            new_sequence = prev_sequence + [new_token]
            # Get the score for this sequence
            new_score = top_scores[i].item()
            # Add to candidates
            new_candidates.append((new_sequence, new_score))
        return new_candidates

    best_seq, best_score = beam_search(
        expand_fn=expand_beam,  # (beam) -> candidates
        initial_beam=[([], 0.0)],
        num_beams=num_beams,
        max_steps=horizon,
    )

    return (
        torch.tensor(best_seq, device=alpha.device),  # (H,)
        torch.tensor(best_score, device=alpha.device),  # (1,)
    )


def sample_from_mps_tensor(
    alpha: torch.Tensor, beta: torch.Tensor, core: torch.Tensor, num_beams: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized beam search for MPS tensor sampling."""
    horizon, _, vocab_size, _ = core.shape
    beam = [([], 0.0)]

    for t in range(horizon):
        if len(beam) == 0:
            break

        # Batch process all sequences in beam
        seqs, log_probs = zip(*beam)
        log_probs = torch.tensor(log_probs, device=alpha.device)  # (beam_size,)

        # Get next token probabilities for all sequences
        # Holds `beam_size` vectors each of dim `vocab_size`. I.e, beam_size x (vocab_size,)
        all_p_next = []
        for seq in seqs:
            p_next, _ = select_margin_mps_tensor(
                alpha=alpha,
                beta=beta,
                core=core,
                ops=torch.tensor(
                    seq + [-1] + [-2] * (horizon - t - 1), device=alpha.device
                ),
                use_scale_factors=False,
            )
            all_p_next.append(p_next)

        all_p_next = torch.stack(all_p_next)  # (beam_size, vocab_size)
        log_p_next = torch.log(all_p_next)

        # Calculate scores for all possible next tokens
        candidate_scores = (
            log_probs.unsqueeze(1) + log_p_next
        )  # (beam_size, vocab_size)

        # Get top-k scores and indices
        flat_scores = candidate_scores.view(-1)
        top_k_scores, top_k_indices = torch.topk(
            flat_scores, k=min(num_beams, len(flat_scores)), dim=0
        )

        # Convert flat indices to sequence and token indices
        prev_seq_indices = top_k_indices // vocab_size
        token_indices = top_k_indices % vocab_size

        # Build new beam
        beam = []
        for i in range(len(top_k_scores)):
            prev_seq = list(seqs[prev_seq_indices[i]])
            new_seq = prev_seq + [token_indices[i].item()]
            beam.append((new_seq, top_k_scores[i].item()))

    return torch.tensor(beam[0][0], device=alpha.device), torch.tensor(
        beam[0][1], device=alpha.device
    )


def sample_from_mps_tensor_old(
    alpha: torch.Tensor, beta: torch.Tensor, core: torch.Tensor, beam_size: int = 1
) -> torch.Tensor:
    """Samples from an MPS tensor representation of probabilities.

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (R)
        beta (torch.Tensor): Beta tensor of shape (R)
        core (torch.Tensor): Core tensor of shape (H, R, D, R)

    Returns:
        torch.Tensor: Sampled tensor of shape (H,)
    """
    selected_indices = []
    horizon, _, _, _ = core.shape
    for t in range(horizon):
        # Unnormalized P(y_t | y_{<t})
        p_tilde_yt_given_prev, _ = select_margin_mps_tensor(
            alpha=alpha,
            beta=beta,
            core=core,
            ops=torch.tensor(
                selected_indices + [-1] + [-2] * (horizon - t - 1),
                device=alpha.device,
            ),
        )  # (D,)
        # Sample from P(y_t | y_{<t})
        selected_index = torch.multinomial(p_tilde_yt_given_prev, num_samples=1).item()
        selected_indices.append(selected_index)
    return torch.tensor(selected_indices, device=alpha.device)


def select_margin_mps_tensor(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    ops: torch.Tensor,
    use_scale_factors: bool = True,
):
    """Performs selection and marginalization operations on a MPS tensor representation.

    Args:
        alpha (torch.Tensor): Alpha tensor of shape (R)
        beta (torch.Tensor): Beta tensor of shape (R)
        core (torch.Tensor): Core tensor of shape (H, R, D, R)
        ops (torch.Tensor): Operation codes of shape (T,) specifying:
            -2: marginalize mode (sum reduction)
            -1: keep mode as free index
            [0,V): select index v in mode

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Result tensor of shape (F, D) where F is the numbe of free indices (-1 operations) in ops
            - Scale factors of shape (T)
    """
    # Validation:
    assert len(core.shape) == 4, "Core tensor must be 4D (non-batched)"
    assert len(ops.shape) == 1, "Ops tensor must be 1D (non-batched)"
    assert (ops >= -2).all() and (ops < core.size(2)).all(), "Invalid ops tensor"

    # Note ops must be in the order of select, free, marginalize
    bp_free, bp_margin = get_breakpoints(
        ops.reshape(1, -1)
    )  # (1,), (1,) selects index at which selects end
    bp_free, bp_margin = int(bp_free.item()), int(bp_margin.item())
    assert bp_free < bp_margin, "Invalid ops tensor (select/marginalize order)"

    scale_factors = []
    # 1. Reduce via selection
    horizon, rank_size, vocab_size, _ = core.shape
    result_select = None
    if bp_free > 0:
        result_select = torch.gather(
            core[:bp_free].permute(0, 1, 3, 2).reshape(-1, vocab_size),
            dim=1,
            index=ops[:bp_free]
            .reshape(-1, 1, 1)
            .repeat(1, rank_size, rank_size)
            .reshape(-1, 1),
        ).reshape(
            bp_free, rank_size, rank_size
        )  # (H', R, D, R) -> (H'RR, D) -> (H'RR, 1) -> (H', R, R)
        if use_scale_factors:
            _scale_factors = torch.linalg.norm(result_select, dim=(-2, -1))
            result_select = result_select / _scale_factors.reshape(-1, 1, 1)
            scale_factors.extend(_scale_factors.tolist())
        result_select = (
            torch.linalg.multi_dot([t for t in result_select])
            if result_select.size(0) > 1
            else result_select.squeeze(0)
        )  # (R, R) x H' => (R, R)

    # 2. Reduce via marginalization
    result_margin = None
    if bp_margin < horizon:
        result_margin = core[bp_margin:].sum(dim=2)  # (H'', R, R)
        if use_scale_factors:
            _scale_factors = torch.linalg.norm(result_margin, dim=(-2, -1))
            result_margin = result_margin / _scale_factors.reshape(-1, 1, 1)
            scale_factors.extend(_scale_factors.tolist())
        result_margin = (
            torch.linalg.multi_dot([t for t in result_margin])
            if result_margin.size(0) > 1
            else result_margin.squeeze(0)
        )  # (R, R) x H'' => (R, R)

    # 3. Combine results
    return (
        mps_to_tensor(
            alpha=alpha @ result_select if result_select is not None else alpha,
            beta=result_margin @ beta if result_margin is not None else beta,
            core=core[bp_free:bp_margin],
        ),
        scale_factors,
    )
