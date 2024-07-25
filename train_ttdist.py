import torch
from TJDNet import sample_from_tensor_dist
from TJDNet.TJDLayer.TTDist import TTDist


def normalize_matrix(matrix):
    """Placeholder normalization function, replace with the actual implementation."""
    norm_factor = torch.norm(matrix, dim=0, keepdim=True)
    return matrix / norm_factor


def make_batched_alpha_beta_core(alpha, beta, core, batch_size):
    alpha = alpha.repeat(batch_size, 1)
    beta = beta.repeat(batch_size, 1)
    core = core.repeat(batch_size, 1, 1, 1)
    return alpha, beta, core


def compute_log_prob_and_norm(alpha, beta, core, target):
    ttdist = TTDist(alpha, beta, core, target.size(1), repeat_batch_size=target.size(0))
    probs_tilde, norm_constant = ttdist.get_prob_and_norm(target)
    loss = (-torch.log(probs_tilde) + torch.log(norm_constant)).mean()
    return loss


def main():
    n_epochs = 500
    batch_size = 8
    n_train_samples = 8 * 100
    n_test_samples = 8 * 10
    true_rank = 4
    model_rank = 2
    vocab_size = 4
    seq_len = 4
    lr = 1e-3

    true_alpha = torch.randn(1, true_rank)
    true_beta = torch.randn(1, true_rank)
    true_core = torch.randn(1, true_rank, vocab_size, true_rank)
    true_ttdist = TTDist(
        true_alpha, true_beta, true_core, seq_len, repeat_batch_size=batch_size
    )
    true_dist = true_ttdist.materialize().squeeze()
    true_dist = true_dist / true_dist.sum()  # P(d1, d2, ..., dN)

    # Sample `batch_size` random samples from the true distribution
    train_samples = sample_from_tensor_dist(true_dist[0], n_train_samples)
    train_dataset = torch.utils.data.TensorDataset(train_samples)  # type: ignore
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # type: ignore

    # Sample `batch_size` random samples from the true distribution
    test_samples = sample_from_tensor_dist(true_dist[0], n_test_samples)

    alpha = torch.nn.Parameter(torch.randn(1, model_rank))
    beta = torch.nn.Parameter(torch.randn(1, model_rank))
    core = torch.nn.Parameter(torch.randn(1, model_rank, vocab_size, model_rank))

    optimizer = torch.optim.Adam([alpha, beta, core], lr=lr)

    for ep in range(n_epochs):
        # Train
        for i, batch in enumerate(train_dataloader):
            target = batch[0]
            optimizer.zero_grad()
            loss = compute_log_prob_and_norm(alpha, beta, core, target)
            loss.backward()
            optimizer.step()

        # Test
        test_loss = compute_log_prob_and_norm(alpha, beta, core, test_samples)
        test_loss_gt = compute_log_prob_and_norm(
            true_alpha, true_beta, true_core, test_samples
        )
        print(
            f"[Epoch {ep}] Loss: {test_loss.item()} | GT Loss: {test_loss_gt.item()} | Loss Diff (abs): {abs(test_loss.item() - test_loss_gt.item())}"
        )


if __name__ == "__main__":
    main()
