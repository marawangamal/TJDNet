import unittest
import math
import torch
from TJDNet import MPSDist
from TJDNet.loss import get_entropy_loss, get_entropy_loss_stable

# Set all random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TestTTDist(unittest.TestCase):

    def test_get_entropy_loss_stable(self):

        # ttdist: MPSDist,
        # samples: torch.Tensor,
        # eps: float = 1e-6,

        vocab_size = 4
        rank = 3
        true_rank = 3
        output_size = 5
        batch_size = 1
        true_dist_init_method = "sparse"

        # 1. Get the true distribution
        true_mpsdist = MPSDist(
            n_vocab=vocab_size,
            rank=true_rank,
            init_method=true_dist_init_method,
        )

        learned_mpsdist = MPSDist(
            n_vocab=vocab_size,
            rank=rank,
        )

        samples = true_mpsdist.sample(
            n_samples=batch_size,
            max_len=output_size,
        ).detach()

        loss_entropy = get_entropy_loss(learned_mpsdist, samples)
        loss_entropy_stable = get_entropy_loss_stable(learned_mpsdist, samples)

        # Expected output to be close
        self.assertTrue(
            math.isclose(loss_entropy.item(), loss_entropy_stable.item(), abs_tol=0.01)
        )


if __name__ == "__main__":
    unittest.main()
