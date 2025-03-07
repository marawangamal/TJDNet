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
