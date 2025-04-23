import unittest
import torch

from tjdnet.tensorops.mps import select_margin_mps_tensor_batched


class TestMPSTensorOps(unittest.TestCase):
    def test_select_margin_mps_tensor_batched_shape(self):
        batch_size, rank, horizon, vocab_size = 2, 2, 4, 3
        # cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (R, T, D)
        alpha = torch.ones(batch_size, rank)
        beta = torch.ones(batch_size, rank)
        core = torch.ones(batch_size, horizon, rank, vocab_size, rank)

        # Create ops tensor with all three operation types:
        # [0, -1, -1, -2] means:
        # - select index 0 in first position
        # - marginalize last two positions
        ops = torch.tensor([[0, -1, -2, -2], [0, 1, -1, -2]])
        result, _ = select_margin_mps_tensor_batched(
            alpha, beta, core, ops
        )  # (rank, n_free, vocab_size)

        # Assert: shape
        self.assertEqual(result.shape, (batch_size, vocab_size))

    def test_select_margin_mps_tensor_batched_values(self):
        batch_size, rank, horizon, vocab_size = 2, 2, 4, 3
        # cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (R, T, D)
        alpha = torch.ones(batch_size, rank)
        beta = torch.ones(batch_size, rank)
        core = torch.ones(batch_size, horizon, rank, vocab_size, rank)

        # Create ops tensor with all three operation types:
        # [0, -1, -1, -2] means:
        # - select index 0 in first position
        # - marginalize last two positions
        ops = torch.tensor([[0, -1, -2, -2], [0, 1, -1, -2]])
        result, scale_factors = select_margin_mps_tensor_batched(
            alpha, beta, core, ops
        )  # (rank, n_free, vocab_size)
        if len(scale_factors):
            result = result * torch.stack(scale_factors, dim=-1).prod(
                dim=-1, keepdim=True
            )

        self.assertTrue(
            torch.allclose(
                result, torch.tensor([[288.0, 288.0, 288.0], [96.0, 96.0, 96.0]])
            )
        )

    def test_select_margin_mps_tensor_batched_orders(self):
        batch_size, rank, horizon, vocab_size = 128, 2, 3, 50256
        # cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (R, T, D)
        alpha = torch.ones(batch_size, rank)
        beta = torch.ones(batch_size, rank)
        core = torch.exp(torch.randn(batch_size, horizon, rank, vocab_size, rank))

        ops_select = torch.randint(0, vocab_size, (batch_size, horizon))
        ops_margin = torch.full((batch_size, horizon), -2, dtype=torch.int64)
        y_tilde, _ = select_margin_mps_tensor_batched(
            alpha, beta, core, ops_select, use_scale_factors=False
        )  # (batch_size,)
        z_tilde, _ = select_margin_mps_tensor_batched(
            alpha, beta, core, ops_margin, use_scale_factors=False
        )

        # Should have that y_tilde >= z_tilde
        self.assertTrue(
            torch.all(y_tilde <= z_tilde),
            msg="y_tilde should be greater than or equal to z_tilde",
        )


if __name__ == "__main__":
    unittest.main()
