import unittest
import torch

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cpb import CPCond
from tjdnet.distributions.tpnet import TensorParamNetConfig


class TestCPBDist(unittest.TestCase):

    # def test_log_prob_match(self):
    #     batch_size, vocab_size, rank, horizon, n_embd = 8, 128, 8, 2, 256
    #     eps = 1e-9

    #     model_head = CPBDist(
    #         BaseDistConfig(
    #             vocab_size=vocab_size,
    #             horizon=horizon,
    #             rank=rank,
    #             param_net=TensorParamNetConfig(
    #                 in_dim=n_embd,
    #             ),
    #         )
    #     )

    #     x = torch.randn(batch_size, n_embd)  # (B, T, D)
    #     y = torch.randint(0, vocab_size, (batch_size, horizon))
    #     log_prob = model_head.log_prob(x=x, y=y)
    #     log_prob_unstable = model_head.log_prob_unstable(x=x, y=y)
    #     # Should be close to each other
    #     assert torch.allclose(log_prob, log_prob_unstable, atol=eps)

    def test_loss(self):
        batch_size, seq_len, vocab_size, rank, horizon, n_embd = 8, 8, 128, 8, 2, 256
        eps = 1e-9

        model_head = CPCond(
            BaseDistConfig(
                vocab_size=vocab_size,
                horizon=horizon,
                rank=rank,
                param_net=TensorParamNetConfig(
                    in_dim=n_embd,
                ),
            )
        )

        last_hidden_state = torch.randn(batch_size, seq_len, n_embd)  # (B, T, D)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len, horizon))
        loss = model_head.forward(
            x=last_hidden_state.reshape(-1, n_embd),
            y=targets.reshape(-1, horizon),
        )
        assert loss.min() > 0

    def test_sample(self):
        batch_size, seq_len, vocab_size, rank, horizon, n_embd = 32, 8, 128, 3, 2, 256
        eps = 1e-9

        model_head = CPCond(
            BaseDistConfig(
                vocab_size=vocab_size,
                horizon=horizon,
                rank=rank,
                param_net=TensorParamNetConfig(
                    in_dim=n_embd,
                ),
            )
        )

        last_hidden_state = torch.randn(batch_size, seq_len, n_embd)  # (B, T, D)
        y, py = model_head.sample(
            x=last_hidden_state,
            horizon=horizon,
            do_sample=True,
            top_k=200,
        )
        assert y.shape == (batch_size, horizon)
        assert py.shape == (batch_size, horizon, vocab_size)
        assert torch.allclose(py.sum(dim=-1), torch.ones_like(py.sum(dim=-1)))


if __name__ == "__main__":
    unittest.main()
