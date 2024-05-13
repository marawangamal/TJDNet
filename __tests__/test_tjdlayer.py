import unittest
import torch
from TJDNet import TJDLayer


class TestTTDist(unittest.TestCase):
    def setUp(self):

        self.emb_size = 32
        self.rank = 1
        self.vocab_size = 100
        self.batch_size = 2
        self.n_core_repititions = 1
        self.tjd_layer = TJDLayer(emb_size=32, rank=1, vocab_size=100, mode="tjd")

    def test_loss(self):

        alpha = torch.ones(self.batch_size, self.rank)
        beta = torch.ones(self.batch_size, self.rank)
        core_a = torch.zeros(self.batch_size, self.rank, self.vocab_size, self.rank)
        core_b = torch.zeros(self.batch_size, self.rank, self.vocab_size, self.rank)
        target_a = torch.randint(
            0, self.vocab_size, (self.batch_size, self.n_core_repititions)
        )
        target_b = torch.randint(
            0, self.vocab_size, (self.batch_size, self.n_core_repititions)
        )

        for b, _target in enumerate(target_a):
            core_a[b, :, _target, :] = 1

        for b, _target in enumerate(target_b):
            core_b[b, :, _target, :] = 1

        loss_a = self.tjd_layer._compute_loss(
            alpha=alpha, beta=beta, core=core_a, target=target_a
        )

        loss_b = self.tjd_layer._compute_loss(
            alpha=alpha, beta=beta, core=core_a, target=target_b
        )

        # Expect the loss a to be lt loss b because the core is 1 at the target index
        self.assertTrue((loss_a < loss_b).all())

        # Expect the loss to be 0 because the core is 1 at the target index
        self.assertEqual(loss_a, 0)


if __name__ == "__main__":
    unittest.main()
