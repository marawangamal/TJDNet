import unittest
import torch

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cp_condl import CPCondl
import tntorch as tn


class TestCPDist(unittest.TestCase):

    def test_cp_condl_log_prob(self):
        B, H, R, D, V = 1, 2, 8, 4, 5
        cp_condl = CPCondl(
            BaseDistConfig(horizon=H, rank=R, embedding_dim=D, vocab_size=V)
        )
        dims = [V] * H

        # Diagonal tensor
        tens_cp = tn.randn(dims, ranks_cp=R)
        tens_cp.cores = [torch.sigmoid(torch.randn(V, R)) for _ in range(H)]

        tens_cp_normalized = tn.randn(dims, ranks_cp=R)
        alpha = torch.softmax(torch.randn(R), dim=-1)
        tens_cp_normalized.cores = [
            # (V, R) * (1, R) -> (V, R)
            (
                torch.softmax(c, dim=0) * alpha.unsqueeze(0)
                if i == 0
                else torch.softmax(c, dim=0)
            )
            for i, c in enumerate(tens_cp.cores)
        ]

        # idx = tuple([1] * H)
        for idx in [tuple([1] * (H - 1) + [0]), tuple([1] * H)]:
            cores_cp = [ti.T for ti in tens_cp.cores]
            y = torch.tensor([idx])

            # log_prob_hat_unst = cp_condl._compute_log_prob_unstable(
            #     alpha_tilde=alpha.reshape(1, -1).repeat(B, 1),
            #     # (B, H, R, V)
            #     p_dists_tilde=torch.stack(cores_cp, dim=0).unsqueeze(0),
            #     y=y,
            # )  # (B,)

            log_prob_hat = cp_condl._compute_log_prob(
                alpha_tilde=alpha.reshape(1, -1).repeat(B, 1),
                # (B, H, R, V)
                p_dists_tilde=torch.stack(cores_cp, dim=0).unsqueeze(0),
                y=y,
            )  # (B,)

            log_prob_actual = torch.log(tens_cp_normalized[idx]).reshape(1)
            torch.testing.assert_close(
                log_prob_actual, log_prob_hat, atol=1e-1, rtol=1e-1
            )


if __name__ == "__main__":
    unittest.main()
