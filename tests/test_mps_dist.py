import unittest
import torch
from TJDNet import (
    MPSDistBase,
)


class TestTTDist(unittest.TestCase):

    def test_mps_dist__shape(self):
        batch_size = 1
        rank = 8
        vocab_size = 10
        alpha = torch.randn(batch_size, rank)
        beta = torch.randn(batch_size, rank)
        core = torch.randn(batch_size, rank, vocab_size, rank)
        mps_dist = MPSDistBase(alpha=alpha, beta=beta, core=core)
        sample = mps_dist.sample(max_len=10)

        # Test shape
        self.assertEqual(tuple(sample.shape), (batch_size, 10), "Incorrect shape")

    def test_mps_dist__values(self):
        # BUG: the last index always is the most probable one?
        batch_size = 1
        rank = 8
        vocab_size = 10
        alpha = torch.randn(batch_size, rank)
        beta = torch.randn(batch_size, rank)
        core = torch.randn(batch_size, rank, vocab_size, rank)
        mps_dist = MPSDistBase(alpha=alpha, beta=beta, core=core)
        sample = mps_dist.sample(max_len=10)

        # expect no more than 3 of the same value
        self.assertTrue(
            (sample[0, 1:] != sample[0, :-1]).sum() >= 7, "Too many repeated values"
        )


if __name__ == "__main__":
    unittest.main()
