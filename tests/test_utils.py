import unittest
import torch
from TJDNet.utils import (
    umps_select_marginalize_batched,
    window_input_ids,
)


class TestTTDist(unittest.TestCase):

    def test_umps_select_marginalize_batched__shape(self):
        rank = 3
        vocab_size = 4
        batch_size = 1
        alpha = torch.randn(batch_size, rank)
        beta = torch.randn(batch_size, rank)
        core = torch.randn(batch_size, rank, vocab_size, rank)

        operation_map = torch.tensor([[1, -1, -1]])

        result, _ = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            operation_map=operation_map,
        )
        expected_shape = (batch_size,)
        self.assertEqual(tuple(result.shape), expected_shape)

    def test_umps_select_marginalize_batched__values(self):
        rank = 3
        vocab_size = 4
        batch_size = 1
        target_val = 44
        target_idx = 2
        target_batch_idx = 0
        n_core_reps = 3
        alpha = torch.eye(batch_size, rank)
        beta = torch.eye(batch_size, rank)
        core = torch.zeros(batch_size, rank, vocab_size, rank)
        rank_divisor = torch.prod(torch.tensor((n_core_reps - 1) * [rank]))
        core[target_batch_idx, :, target_idx, :] = torch.ones(rank) * (
            target_val / rank_divisor
        ) ** (1 / n_core_reps)

        operation_map = torch.tensor(batch_size * [n_core_reps * [target_idx]])
        result, _ = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            operation_map=operation_map,
            apply_scale_factors=True,
        )
        self.assertAlmostEqual(target_val, result[target_batch_idx].item(), places=3)

    def test_window_input_ids_single_sequence(self):
        # Sample input_ids with a single sequence
        input_ids = torch.tensor([[1, 2, 3, 4]])

        # Define H
        H = 2

        # Expected output
        expected_output = torch.tensor([[[2, 3], [3, 4], [4, 0], [0, 0]]])

        # Run the function
        output = window_input_ids(input_ids, H)

        # Test the result
        assert torch.equal(
            output, expected_output
        ), f"Expected {expected_output}, but got {output}"

    def test_window_input_ids_large_horizon(self):
        # Sample input_ids
        input_ids = torch.tensor([[1, 2, 3]])

        # Define H larger than sequence length
        H = 5

        # Expected output
        expected_output = torch.tensor(
            [[[2, 3, 0, 0, 0], [3, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
        )

        # Run the function
        output = window_input_ids(input_ids, H)

        # Test the result
        assert torch.equal(
            output, expected_output
        ), f"Expected {expected_output}, but got {output}"

    def test_window_input_ids(self):
        # Sample input_ids
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        # Define H
        H = 3

        # Expected output
        expected_output = torch.tensor(
            [
                [[2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 0], [0, 0, 0]],
                [[7, 8, 9], [8, 9, 10], [9, 10, 0], [10, 0, 0], [0, 0, 0]],
            ]
        )

        # Run the function
        output = window_input_ids(input_ids, H)

        # Test the result
        assert torch.equal(
            output, expected_output
        ), f"Expected {expected_output}, but got {output}"


if __name__ == "__main__":
    unittest.main()
