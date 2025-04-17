import unittest
import torch

from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.mps import MPSDist
from tjdnet.distributions.tpnet import TensorParamNetConfig


class TestMPSDist(unittest.TestCase):
    def test_select_from_cp_tensor(self):
        batch_size, seq_len, vocab_size, rank, horizon, n_embd = 8, 8, 128, 8, 2, 256
        eps = 1e-9

        model_head = MPSDist(
            BaseDistConfig(
                vocab_size=vocab_size,
                horizon=horizon,
                rank=rank,
                param_net=TensorParamNetConfig(
                    in_dim=n_embd,
                    out_dim_encoder=horizon * rank * rank,
                    out_dim_decoder=vocab_size,
                ),
            )
        )

        last_hidden_state = torch.randn(batch_size, seq_len, n_embd)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len, horizon))

        p_tilde, p_tilde_scale_factors, norm_const, norm_const_scale_factors = (
            model_head.evaluate_at_points_and_get_norm_consts(
                last_hidden_state, targets
            )
        )  # (B, T-H)

        loss = (
            -torch.log(p_tilde + eps)  # (B, T')
            + torch.log(norm_const)  # (B, T')
            # Contraction Stability Scale Factors
            - sum([torch.log(z) for z in p_tilde_scale_factors])  # (B, T')
            + sum([torch.log(z) for z in norm_const_scale_factors])
        )  # (B, T-H)

        loss = loss.sum(dim=-1).mean()
        self.assertLess(loss.item(), 1e3)

        # Loss should be non-negative
        self.assertGreaterEqual(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
