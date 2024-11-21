import unittest
import torch

from utils.tensorops.cp import select_from_cp_tensor, sum_cp_tensor


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


if __name__ == "__main__":
    unittest.main()
