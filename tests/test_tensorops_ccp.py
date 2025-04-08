import unittest
import torch

from tjdnet.tensorops.ccp import select_margin_ccp_tensor_batched
from tjdnet.tensorops.common import get_breakpoints
from tjdnet.tensorops.cp import (
    select_from_cp_tensor,
    select_margin_cp_tensor,
    select_margin_cp_tensor_batched,
    sum_cp_tensor,
)


class TestCPTensor(unittest.TestCase):
    def test_select_margin_cp_tensor_batched_shape(self):
        batch_size, rank, seq_len, vocab_size = 2, 2, 4, 3
        cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (R, T, D)

        # Create ops tensor with all three operation types:
        # [0, -1, -1, -2] means:
        # - select index 0 in first position
        # - marginalize last two positions
        ops = torch.tensor([[0, -1, -2, -2], [0, 1, -1, -2]])
        result, _ = select_margin_ccp_tensor_batched(
            cp_params,
            cp_decode=torch.eye(vocab_size, vocab_size),
            ops=ops,
        )  # (rank, n_free, vocab_size)

        # Assert: shape
        self.assertEqual(result.shape, (batch_size, vocab_size))

    def test_select_margin_cp_tensor_batched_values(self):
        batch_size, rank, seq_len, vocab_size = 2, 2, 4, 3
        cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (B, R, T, D)

        # Create ops tensor with all three operation types:
        # [0, -1, -1, -2] means:
        # - select index 0 in first position
        # - marginalize last two positions
        ops = torch.tensor([[0, -1, -2, -2], [0, 1, -1, -2]])
        p_tilde, p_tilde_sfs = select_margin_ccp_tensor_batched(
            cp_params,
            cp_decode=torch.eye(vocab_size, vocab_size),
            ops=ops,
        )  # (B, V), (B,)
        sf = torch.stack(p_tilde_sfs, dim=-1).prod(dim=-1)  # (B,)

        self.assertTrue(
            torch.allclose(
                p_tilde * sf.unsqueeze(-1),
                torch.tensor([[18.0, 18.0, 18.0], [6.0, 6.0, 6.0]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
