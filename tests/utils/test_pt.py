import unittest
import torch


def logsumexpsparse_vec(x: torch.Tensor) -> torch.Tensor:
    x_vals = x.coalesce().values()
    c = x_vals.max()
    n_zeros = x.size(-1) - x_vals.size(-1)

    term_a = torch.sum(torch.exp(x_vals - c))
    term_b = n_zeros * torch.exp(-c)
    return c + torch.log(term_a + term_b)


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


if __name__ == "__main__":
    unittest.main()
