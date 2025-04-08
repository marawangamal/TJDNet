import unittest
import torch
from tjdnet.tensorops.common import (
    batch_multi_dim_index,
    get_flat_index,
    get_inactive_indices,
    pop_tensor,
)


class TestTTDist(unittest.TestCase):

    def test_get_flat_index_1d_tensor(self):
        shape = (5,)
        indices = torch.tensor([3])
        result = get_flat_index(indices, shape)
        self.assertEqual(result, 3)

    def test_get_flat_index_2d_tensor(self):
        shape = (3, 4)
        indices = torch.tensor([1, 2])
        result = get_flat_index(indices, shape)
        self.assertEqual(result, 6)

    def test_batch_multi_dim_index_2d_tensor(self):
        tens = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        indices = torch.tensor([[0], [2]])  # (2, 1)
        result = batch_multi_dim_index(tens, indices)
        expected = torch.tensor([1, 6])
        self.assertTrue(torch.all(result == expected))

    def test_batch_multi_dim_index_3d_tensor(self):
        tens = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
        indices = torch.tensor([[0, 1], [1, 0]])  # (2, 2)
        result = batch_multi_dim_index(tens, indices)
        expected = torch.tensor([2, 7])
        self.assertTrue(torch.all(result == expected))

    def test_pop_tensor(self):
        # set seed for reproducibility
        torch.manual_seed(0)
        batch_size, seq_len = 30, 5
        stop_token_id = 100
        output_seqs = torch.randint(0, 100, (batch_size, seq_len))
        output_seqs[:10, -1] = stop_token_id

        inactive_indices = get_inactive_indices(output_seqs, stop_token_id)
        output_seqs_active, popped_tensors = pop_tensor(
            output_seqs, indices=inactive_indices
        )

        # 1. Check inactive indices == 0:10
        expected_inactive_indices = torch.arange(10)
        self.assertTrue(torch.all(inactive_indices == expected_inactive_indices))

        # 2. Check that output_seqs_active is same as output_seqs[10:]
        self.assertTrue(torch.all(output_seqs_active == output_seqs[10:]))


if __name__ == "__main__":
    unittest.main()
