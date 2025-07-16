import unittest
import torch

from tjdnet.tensorops.common import get_breakpoints
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched
import tntorch as tn


class TestCPTensor(unittest.TestCase):

    def test_get_breakpoints(self):
        ops = torch.tensor([[0, -1, -2, -2], [0, 0, -1, -2]])  # (B, T)
        bp_free, bp_margin = get_breakpoints(ops)

        # Assert: bp_free value
        self.assertEqual(bp_free.shape, (2,))
        self.assertTrue(torch.allclose(bp_free, torch.tensor([1, 2])))

        # Assert: bp_margin value
        self.assertEqual(bp_margin.shape, (2,))
        self.assertTrue(torch.allclose(bp_margin, torch.tensor([2, 3])))

    def test_select_margin_cp_tensor_batched_shape(self):
        batch_size, rank, seq_len, vocab_size = 2, 2, 4, 3
        cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (R, T, D)

        # Create ops tensor with all three operation types:
        # [0, -1, -1, -2] means:
        # - select index 0 in first position
        # - marginalize last two positions
        ops = torch.tensor([[0, -1, -2, -2], [0, 1, -1, -2]])
        result, _ = select_margin_cp_tensor_batched(
            cp_params, ops
        )  # (rank, n_free, vocab_size)

        # Assert: shape
        self.assertEqual(result.shape, (batch_size, vocab_size))

    def test_select_margin_cp_tensor_batched_values(self):
        # batch_size, rank, seq_len, vocab_size = 2, 2, 4, 3
        # cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (R, T, D)
        B, H, R, V = 1, 3, 2, 5
        dims = [V] * H
        # Random tensor
        full = torch.randn(dims).sigmoid()
        full = full / full.sum()

        t = tn.Tensor(full, ranks_cp=R, verbose=True)
        y = torch.randint(0, V, (B, H))  # (B, H)
        cores = [c.T for c in t.cores]  # (R, V)
        res_hat, _ = select_margin_cp_tensor_batched(
            # (B, R, H, V)
            cp_params=torch.stack(cores, dim=1).unsqueeze(0),
            ops=y,
            use_scale_factors=False,
        )  # (B, V)
        res_actual = torch.stack([t[tuple(yi.tolist())] for yi in y], dim=0)
        torch.testing.assert_close(res_hat, res_actual)

    # def test_select_margin_cp_tensor_batched_values(self):
    #     batch_size, rank, seq_len, vocab_size = 2, 2, 4, 3
    #     cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (R, T, D)

    #     # Create ops tensor with all three operation types:
    #     # [0, -1, -1, -2] means:
    #     # - select index 0 in first position
    #     # - marginalize last two positions
    #     ops = torch.tensor([[0, -1, -2, -2], [0, 1, -1, -2]])
    #     result_batched, _ = select_margin_cp_tensor_batched(
    #         cp_params, ops
    #     )  # (rank, n_free, vocab_size)

    #     self.assertTrue(
    #         torch.allclose(
    #             result_batched, torch.tensor([[18.0, 18.0, 18.0], [6.0, 6.0, 6.0]])
    #         )
    #     )


if __name__ == "__main__":
    unittest.main()
