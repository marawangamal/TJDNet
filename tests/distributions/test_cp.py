import unittest
import torch

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.tpnet import TensorParamNetConfig


class TestCPDist(unittest.TestCase):
    def test_select_from_cp_tensor(self):
        batch_size, vocab_size, rank, horizon, n_embd = 8, 128, 8, 2, 256
        eps = 1e-9

        model_head = CPDist(
            BaseDistConfig(
                vocab_size=vocab_size,
                horizon=horizon,
                rank=rank,
                param_net=TensorParamNetConfig(
                    in_dim=n_embd,
                ),
            )
        )

        last_hidden_state = torch.randn(batch_size, n_embd)
        targets = torch.randint(0, vocab_size, (batch_size, horizon))

        p_tilde, p_tilde_scale_factors, norm_const, norm_const_scale_factors = (
            model_head.evaluate(x=last_hidden_state, y=targets)
        )  # (B, T-H)

        loss = (
            -torch.log(p_tilde + eps)  # (B,)
            + torch.log(norm_const)  # (B,)
            # Contraction Stability Scale Factors
            - sum([torch.log(z) for z in p_tilde_scale_factors])  # (B,)
            + sum([torch.log(z) for z in norm_const_scale_factors])
        )  # (B, T-H)

        loss = loss.sum(dim=-1).mean()
        self.assertLess(loss.item(), 1e3)

    def test_memory_usage(self):
        batch_size, seq_len, vocab_size, rank, horizon, n_embd = 8, 128, 128, 8, 2, 256
        bt = batch_size * seq_len

        model_head = CPDist(
            BaseDistConfig(
                vocab_size=vocab_size,
                horizon=horizon,
                rank=rank,
                param_net=TensorParamNetConfig(
                    in_dim=n_embd,
                ),
            )
        )
        last_hidden_state = torch.randn(bt, n_embd)
        targets = torch.randint(0, vocab_size, (bt, horizon))

        loss = model_head.forward(
            x=last_hidden_state,
            y=targets,
        )


if __name__ == "__main__":
    unittest.main()
