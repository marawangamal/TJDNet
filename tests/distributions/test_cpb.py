import unittest
import torch

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cpb_v2 import CPBDist
from tjdnet.distributions.tpnet import TensorParamNetConfig


class TestCPDist(unittest.TestCase):
    def test_select_from_cp_tensor(self):
        B, V, R, H, D = 8, 128, 8, 2, 256
        eps = 1e-9

        model_head = CPBDist(
            BaseDistConfig(
                vocab_size=V,
                horizon=H,
                rank=R,
                param_net=TensorParamNetConfig(
                    in_dim=D,
                ),
            )
        )

        x = torch.randn(B, D)
        y = torch.randint(0, V, (B, H))
        loss = model_head.log_prob(x, y)
        self.assertTrue(torch.isfinite(loss).all(), "Loss should be finite")
        self.assertEqual(loss.shape, (B,))


if __name__ == "__main__":
    unittest.main()
