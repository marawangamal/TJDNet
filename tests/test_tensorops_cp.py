import unittest
import torch

from tensorops.cp import select_from_cp_tensor, sum_cp_tensor, select_margin_cp_tensor


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

    def test_select_margin_cp_tensor(self):
        # Create a simple CP tensor with rank 2, sequence length 4, and vocab size 3
        rank, seq_len, vocab_size = 2, 4, 3
        cp_params = torch.ones(rank, seq_len, vocab_size)  # (R, T, D)

        # Create ops tensor with all three operation types:
        # [0, -1, -1, -2] means:
        # - select index 0 in first position
        # - keep next two positions as free indices
        # - marginalize last position
        ops = torch.tensor([0, -1, -1, -2])

        result, _ = select_margin_cp_tensor(
            cp_params, ops
        )  # (rank, n_free, vocab_size)

        # Expected shape should be (rank, n_free, vocab_size)
        # where n_free is number of -1 operations (2 in this case)
        expected_shape = (rank, 2, vocab_size)
        self.assertEqual(result.shape, expected_shape)

        # Since we used all ones and selected index 0:
        # - Select operation gives 1
        # - Free indices remain unchanged
        # - Marginalize sums all values (equals vocab_size)
        # Final result should be ones * vocab_size
        expected_result = torch.ones(expected_shape) * vocab_size
        self.assertTrue(torch.allclose(result, expected_result))


if __name__ == "__main__":
    unittest.main()
