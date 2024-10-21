import unittest
import torch
from TJDNet.tensop import batch_multi_dim_index, get_flat_index


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


if __name__ == "__main__":
    unittest.main()
