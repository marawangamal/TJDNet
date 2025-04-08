import unittest
import torch

from tjdnet.tensorops.common import get_breakpoints
from tjdnet.tensorops.cp import (
    select_from_cp_tensor,
    select_margin_cp_tensor,
    select_margin_cp_tensor_batched,
    sum_cp_tensor,
)


class TestCPTensor(unittest.TestCase):
    def test_select_from_cp_tensor(self):
        batch_size, rank, seq_len, n_embd = 1, 2, 3, 4
        tensor = torch.ones(batch_size, rank, seq_len, n_embd) / 2  # (B, R, T, D)
        indices = torch.randint(0, n_embd, (batch_size, seq_len))
        result = select_from_cp_tensor(tensor, indices)
        self.assertTrue(
            torch.allclose(result, torch.ones_like(result) * 0.25, atol=1e-6)
        )

    def test_sum_cp_tensor(self):
        # Test case 1: Simple tensor with all ones
        tensor = torch.ones(1, 2, 2, 4)  # batch=1, rank=2, seq_len=2, embd=4
        result = sum_cp_tensor(tensor)
        expected = torch.tensor([32.0])  # 1 * 2 * 3 * 4
        self.assertTrue(torch.allclose(result, expected))

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
        batch_size, rank, seq_len, vocab_size = 2, 2, 4, 3
        cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (R, T, D)

        # Create ops tensor with all three operation types:
        # [0, -1, -1, -2] means:
        # - select index 0 in first position
        # - marginalize last two positions
        ops = torch.tensor([[0, -1, -2, -2], [0, 1, -1, -2]])
        result_batched, _ = select_margin_cp_tensor_batched(
            cp_params, ops
        )  # (rank, n_free, vocab_size)

        self.assertTrue(
            torch.allclose(
                result_batched, torch.tensor([[18.0, 18.0, 18.0], [6.0, 6.0, 6.0]])
            )
        )

    def test_select_margin_cp_tensor_batched_values__match_select(self):
        batch_size, rank, horizon, vocab_size = 8, 4, 4, 128
        cp_params = torch.ones(batch_size, rank, horizon, vocab_size)  # (B, R, T, D)

        # Create ops tensor with all three operation types:
        # [0, -1, -1, -2] means:
        # - select index 0 in first position
        # - marginalize last two positions
        for _ in range(3):
            ops = torch.randint(0, vocab_size, (batch_size, horizon))
            result_batched, _ = select_margin_cp_tensor_batched(
                cp_params, ops
            )  # (rank, n_free, vocab_size)
            expected_result = select_from_cp_tensor(cp_params, ops)
            self.assertTrue(torch.allclose(result_batched, expected_result))

    # def test_select_margin_cp_tensor_batched_values__match_select_marginalize(self):
    #     # Set seed for reproducibility
    #     torch.manual_seed(0)
    #     batch_size, rank, horizon, vocab_size = 3, 4, 5, 4
    #     # cp_params = torch.rand(batch_size, rank, horizon, vocab_size)  # (B, R, T, D)
    #     cp_params = torch.ones(batch_size, rank, horizon, vocab_size)  # (B, R, T, D)

    #     # Create ops tensor with all three operation types:
    #     # [0, -1, -1, -2] means:
    #     # - select index 0 in first position
    #     # - marginalize last two positions
    #     # [1.0098, 0.9534, 1.4481, 0.2662],
    #     ops = torch.tensor([[0, 1, 2, -1, -2], [0, 1, -1, -2, -2], [0, -1, -2, -2, -2]])
    #     result_batched, _ = select_margin_cp_tensor_batched(
    #         cp_params, ops
    #     )  # (rank, n_free, vocab_size)
    #     expected_result = torch.stack(
    #         [
    #             select_margin_cp_tensor(cp_params[i], ops[i])[0]
    #             for i in range(batch_size)
    #         ]
    #     )
    #     self.assertTrue(torch.allclose(result_batched, expected_result))


if __name__ == "__main__":
    unittest.main()
