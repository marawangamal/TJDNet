import unittest
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched
import torch

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cp import CPDist
import tntorch as tn

# from tjdnet.distributions.tpnet import TensorParamNetConfig


class TestLemmas(unittest.TestCase):
    def test_case_1(self):
        D, R, V = 10, 2, 5
        b = torch.randn(R, D)
        e = torch.randn(D, V)

        a = torch.einsum("rd,dv->rv", b, e)

        y1 = torch.log_softmax(a, dim=-1)
        y2 = a - torch.logsumexp(a, dim=-1, keepdim=True)

        self.assertTrue(torch.allclose(y1, y2))


if __name__ == "__main__":
    unittest.main()
