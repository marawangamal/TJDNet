import unittest
import torch

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cp_condl import CPCondl
from tjdnet.distributions.cp_cond import CPCond


class TestCPDist(unittest.TestCase):
    def test_cp_condl_values(self):
        # Test configuration
        config = BaseDistConfig(
            horizon=3, rank=2, embedding_dim=4, vocab_size=5, positivity_func="sigmoid"
        )

        # Create both distributions
        cp_condl = CPCondl(config)
        cp_cond = CPCond(config)

        # Now, set the same weights for both distributions
        cp_cond_params = dict(cp_cond.named_parameters())
        for name, param in cp_condl.named_parameters():
            if name in cp_cond_params:
                param.data = cp_cond_params[name].data
                print(f"Copied {name} from cp_cond to cp_condl")

        # Set the same random seed for reproducible results
        torch.manual_seed(42)

        # Create test data
        batch_size = 2
        x = torch.randn(batch_size, config.embedding_dim)
        y = torch.randint(0, config.vocab_size, (batch_size, config.horizon))

        # Compute log probabilities using both methods
        log_prob_condl = cp_condl.log_prob(x, y, return_dist_slice=False)
        log_prob_cond = cp_cond.log_prob(x, y, return_dists=False)

        # The results should be equal (within numerical precision)
        torch.testing.assert_close(log_prob_condl, log_prob_cond, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
