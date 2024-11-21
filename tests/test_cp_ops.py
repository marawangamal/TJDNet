import unittest
import torch

from utils.tensorops.cp import select_from_cp_tensor


class TestCPTensor(unittest.TestCase):
    def test_correct_values(self):
        batch_size, rank, seq_len, n_embd = 1, 2, 3, 4
        tensor = torch.ones(batch_size, rank, seq_len, n_embd) / 2  # (B, R, T, D)
        indices = torch.randint(0, n_embd, (batch_size, seq_len))
        result = select_from_cp_tensor(tensor, indices)
        # expect every value to be 1
        self.assertTrue(torch.allclose(result, torch.ones_like(result), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
