import unittest
import math
import torch
from TJDNet.utils import (
    batched_index_select,
    umps_select_marginalize_batched,
)


class TestTTDist(unittest.TestCase):

    def test_batched_index_select(self):
        input_tensor = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
                [[28, 29, 30], [31, 32, 33], [34, 35, 36]],
                [[37, 38, 39], [40, 41, 42], [43, 44, 45]],
            ]
        )  # Shape: (5, 3, 3)

        batched_index = torch.tensor([[4, 2, 1], [0, 0, 0], [0, 2, 1]])

        # Expected output contains the values at the specified indices
        expected_output = torch.tensor([44, 1, 8])

        # Perform batched index select
        result = batched_index_select(input_tensor, batched_index)

        # Check if the output matches the expected output
        assert torch.equal(
            result, expected_output
        ), "Test Failed: Output does not match expected output"
        print("Test Passed: Output matches the expected output")

    def test_umps_select_marginalize_batched__shape(self):
        rank = 3
        vocab_size = 4
        n_core_repititions = 3
        alpha = torch.randn(1, rank)
        beta = torch.randn(1, rank)
        core = torch.randn(1, rank, vocab_size, rank)

        selection_map = torch.tensor([[1, -1, -1]])
        marginalize_mask = torch.tensor([[0, 0, 1]])

        result = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            selection_map=selection_map,
            marginalize_mask=marginalize_mask,
        )
        expected_shape = (1, vocab_size)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(tuple(result.shape), expected_shape)

    def test_umps_select_marginalize_batched__select_only(self):
        rank = 3
        vocab_size = 4
        n_core_repititions = 3
        alpha = torch.randn(1, rank)
        beta = torch.randn(1, rank)
        core = torch.randn(1, rank, vocab_size, rank)

        selection_map = torch.tensor([[1, 1, -1]])
        marginalize_mask = torch.tensor([[0, 0, 0]])

        result = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            selection_map=selection_map,
            marginalize_mask=marginalize_mask,
        )
        expected_shape = (1, vocab_size)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(tuple(result.shape), expected_shape)

    def test_umps_select_marginalize_batched__marginalize_only(self):
        rank = 3
        vocab_size = 4
        n_core_repititions = 3
        alpha = torch.randn(1, rank)
        beta = torch.randn(1, rank)
        core = torch.randn(1, rank, vocab_size, rank)

        selection_map = torch.tensor([[-1, -1, -1]])
        marginalize_mask = torch.tensor([[1, 1, 0]])

        result = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            selection_map=selection_map,
            marginalize_mask=marginalize_mask,
        )
        expected_shape = (1, vocab_size)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(tuple(result.shape), expected_shape)

    def test_umps_select_marginalize_batched__correct_values_select(self):
        rank = 3
        vocab_size = 4
        batch_size = 1
        n_core_repititions = 3
        expected_output = 44
        alpha = torch.eye(1, rank)
        beta = torch.eye(1, rank)
        core = torch.zeros(1, rank, vocab_size, rank)
        for i in range(vocab_size):
            core[0, :, i, :] = torch.eye(rank)

        core[0, 0, 0, 0] = expected_output

        selection_map = torch.tensor([[1, 1, -1]])
        marginalize_mask = torch.tensor([[0, 0, 0]])

        result = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            selection_map=selection_map,
            marginalize_mask=marginalize_mask,
        )

        self.assertEqual(expected_output, result[0, 0].item())

    def test_umps_select_marginalize_batched__correct_values_margin(self):
        rank = 3
        vocab_size = 4
        alpha = torch.ones(1, rank)
        beta = torch.ones(1, rank)
        core = torch.zeros(1, rank, vocab_size, rank)
        for i in range(vocab_size):
            core[0, :, i, :] = torch.eye(rank) * (1 / vocab_size)

        selection_map = torch.tensor([[-1, -1, -1]])
        marginalize_mask = torch.tensor([[0, 1, 1]])

        result = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            selection_map=selection_map,
            marginalize_mask=marginalize_mask,
        )
        self.assertTrue(math.isclose(rank, result.sum().item(), abs_tol=0.001))


if __name__ == "__main__":
    unittest.main()
