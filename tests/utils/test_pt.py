import unittest
import torch

from tjdnet.tensorops.mps import select_margin_mps_tensor_batched


def logsumexpsparse_vec(x: torch.Tensor) -> torch.Tensor:
    x_vals = x.coalesce().values()
    c = x_vals.max()
    n_zeros = x.size(-1) - x_vals.size(-1)

    term_a = torch.sum(torch.exp(x_vals - c))
    term_b = n_zeros * torch.exp(-c)
    return c + torch.log(term_a + term_b)


def get_stats(tensor: torch.Tensor) -> dict:
    """Get summary statistics of a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        dict: Dictionary containing mean, std, min, and max.
    """
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


class TestPT(unittest.TestCase):
    def test_lse(self):
        # Test if we can do torch.logsumexp with sparse tensors

        for i in range(5, 10):
            # Create a sparse tensor
            indices = torch.randint(
                0, i, (1, i - 1)
            )  # First dimension is sparse_dim (1), second is nnz

            # Generate random values
            values = torch.rand(i - 1) * 10.0

            # LogSumExp
            sparse_tensor = torch.sparse_coo_tensor(indices, values, (i,))

            # Perform logsumexp
            result = logsumexpsparse_vec(sparse_tensor)

            # Calculate expected result
            expected = torch.logsumexp(sparse_tensor.to_dense(), dim=0)
            torch.testing.assert_close(result, expected)

    def test_umps_numerical_dist(self):
        # Test if we can do torch.logsumexp with sparse tensors
        batch_size, seq_len, horizon, rank, vocab_size = 1, 9, 1, 8, 1024
        alpha = torch.ones(batch_size, rank)
        beta = torch.ones(batch_size, rank)
        core = torch.randn(batch_size, rank, horizon, rank)

        # Pure select
        ops = torch.randint(0, vocab_size, (batch_size, seq_len))
        p_tilde, sf_p_tilde = select_margin_mps_tensor_batched(
            alpha,
            beta,
            core,
            ops,
        )

        # Print summary stats of p_tilde
        p_tilde_stats = get_stats(p_tilde)
        sf_p_tilde_stats = get_stats(torch.stack(sf_p_tilde, dim=0))
        print("p_tilde stats:")
        for key, value in p_tilde_stats.items():
            print(f"{key}: {value:.4f}")
        print("sf_p_tilde stats:")
        for key, value in sf_p_tilde_stats.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    unittest.main()
