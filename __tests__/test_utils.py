import unittest
import torch
from TJDNet.utils import (
    batched_index_select,
    umps_batch_select_marginalize,
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

    def test_select_and_marginalize_uMPS_1D(self):
        rank = 3
        vocab_size = 4
        output_size = 3
        alpha = torch.randn(rank)
        beta = torch.randn(rank)
        core = torch.randn(rank, vocab_size, rank)

        result = umps_batch_select_marginalize(
            alpha, beta, core, output_size, {0: 1}, [2]
        )
        expected_shape = (vocab_size,)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(tuple(result.shape), expected_shape)

    def test_select_and_marginalize_uMPS_2D(self):
        rank = 3
        vocab_size = 4
        output_size = 4
        alpha = torch.randn(rank)
        beta = torch.randn(rank)
        core = torch.randn(rank, vocab_size, rank)

        result = umps_batch_select_marginalize(
            alpha, beta, core, output_size, {0: 1}, [2]
        )
        expected_shape = (vocab_size, vocab_size)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(tuple(result.shape), expected_shape)

    def test_select_and_marginalize_uMPS_3D(self):
        rank = 3
        vocab_size = 4
        output_size = 5
        alpha = torch.randn(rank)
        beta = torch.randn(rank)
        core = torch.randn(rank, vocab_size, rank)

        result = umps_batch_select_marginalize(
            alpha, beta, core, output_size, {0: 1, 3: 1}, [2]
        )
        expected_shape = (vocab_size, vocab_size)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(tuple(result.shape), expected_shape)


if __name__ == "__main__":
    unittest.main()
