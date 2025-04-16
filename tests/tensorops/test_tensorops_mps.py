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


if __name__ == "__main__":
    unittest.main()
