import unittest

import torch

from TJDNet import create_core_ident, apply_id_transform


class TestCoreIdent(unittest.TestCase):
    def test_identity_matrices(self):
        batch_size = 3
        vocab_size = 5
        rank = 4

        # Generate the core identity tensor
        core_ident = create_core_ident(
            batch_size, vocab_size, rank
        )  # (batch_size, vocab_size, rank, rank)

        # Check the shape of the tensor
        self.assertEqual(core_ident.shape, (batch_size, rank, vocab_size, rank))

        # Check that each slice is an identity matrix
        expected_identity = torch.eye(rank)
        for b in range(batch_size):
            for v in range(vocab_size):
                # Check if the sub-tensor is an identity matrix
                self.assertTrue(
                    torch.all(core_ident[b, :, v, :] == expected_identity).item(),
                    f"Sub-tensor at batch {b} and vocab {v} is not an identity matrix.",
                )

    def test_apply_id_transform(self):
        id_map = {1: 5, 2: 6, 3: 7}
        target = torch.tensor([[0, 1, 2, 3, 4], [2, 2, 1, 3, 0]])

        expected_result = torch.tensor([[0, 5, 6, 7, 4], [6, 6, 5, 7, 0]])
        result = apply_id_transform(target, id_map)

        # Check if the transformed target matches the expected result
        torch.testing.assert_allclose(result, expected_result)  # type: ignore

    def test_empty_map(self):
        id_map = {}
        target = torch.tensor([[0, 1, 2], [2, 1, 0]])

        # The result should be unchanged since the map is empty
        result = apply_id_transform(target, id_map)
        torch.testing.assert_allclose(result, target)  # type: ignore

    def test_no_transformation(self):
        id_map = {5: 10}  # No ID in target matches the map
        target = torch.tensor([[0, 1, 2], [2, 1, 0]])

        # The result should be unchanged since no ID matches
        result = apply_id_transform(target, id_map)
        torch.testing.assert_allclose(result, target)  # type: ignore

    def test_negative_values(self):

        id_map = {-100: 50256}

        target = torch.tensor(
            [
                [
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                ],
                [
                    5199,
                    347,
                    2852,
                    353,
                    796,
                    220,
                    198,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                ],
            ]
        )

        expected_result = torch.tensor(
            [
                [
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                ],
                [
                    5199,
                    347,
                    2852,
                    353,
                    796,
                    220,
                    198,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                    50256,
                ],
            ]
        )

        result = apply_id_transform(target, id_map)
        torch.testing.assert_allclose(result, expected_result)  # type: ignore


if __name__ == "__main__":
    unittest.main()
