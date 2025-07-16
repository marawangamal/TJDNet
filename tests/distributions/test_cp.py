import unittest
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched
import torch

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cp import CPDist
import tntorch as tn

# from tjdnet.distributions.tpnet import TensorParamNetConfig


class TestCPDist(unittest.TestCase):
    # def test_select_from_cp_tensor(self):
    #     batch_size, vocab_size, rank, horizon, n_embd = 8, 128, 8, 2, 256
    #     eps = 1e-9

    #     model_head = CPDist(
    #         BaseDistConfig(
    #             vocab_size=vocab_size,
    #             horizon=horizon,
    #             rank=rank,
    #             param_net=TensorParamNetConfig(
    #                 in_dim=n_embd,
    #             ),
    #         )
    #     )

    #     last_hidden_state = torch.randn(batch_size, n_embd)
    #     targets = torch.randint(0, vocab_size, (batch_size, horizon))

    #     p_tilde, p_tilde_scale_factors, norm_const, norm_const_scale_factors = (
    #         model_head.evaluate(x=last_hidden_state, y=targets)
    #     )  # (B, T-H)

    #     loss = (
    #         -torch.log(p_tilde + eps)  # (B,)
    #         + torch.log(norm_const)  # (B,)
    #         # Contraction Stability Scale Factors
    #         - sum([torch.log(z) for z in p_tilde_scale_factors])  # (B,)
    #         + sum([torch.log(z) for z in norm_const_scale_factors])
    #     )  # (B, T-H)

    #     loss = loss.sum(dim=-1).mean()
    #     self.assertLess(loss.item(), 1e3)

    # def test_memory_usage(self):
    #     batch_size, seq_len, vocab_size, rank, horizon, n_embd = 8, 128, 128, 8, 2, 256
    #     bt = batch_size * seq_len

    #     model_head = CPDist(
    #         BaseDistConfig(
    #             vocab_size=vocab_size,
    #             horizon=horizon,
    #             rank=rank,
    #             param_net=TensorParamNetConfig(
    #                 in_dim=n_embd,
    #             ),
    #         )
    #     )
    #     last_hidden_state = torch.randn(bt, n_embd)
    #     targets = torch.randint(0, vocab_size, (bt, horizon))

    #     loss = model_head.forward(
    #         x=last_hidden_state,
    #         y=targets,
    #     )

    # def test_sample(self):
    #     batch_size, vocab_size, rank, horizon, n_embd = 8, 128, 8, 2, 256

    #     model_head = CPDist(
    #         BaseDistConfig(
    #             vocab_size=vocab_size,
    #             horizon=horizon,
    #             rank=rank,
    #             param_net=TensorParamNetConfig(
    #                 in_dim=n_embd,
    #             ),
    #         )
    #     )

    #     last_hidden_state = torch.randn(batch_size, n_embd)
    #     y_hat, py_tilde = model_head.sample(
    #         x=last_hidden_state,
    #         sample_fn=lambda p: torch.multinomial(p, num_samples=1).squeeze(-1),
    #         horizon=horizon,
    #         return_logits=True,
    #     )

    #     self.assertEqual(y_hat.shape, (batch_size, horizon))
    #     self.assertEqual(py_tilde.shape, (batch_size, horizon, vocab_size))

    def test_cpdist_log_prob(self):
        B, H, R, D, V = 1, 3, 2, 4, 5
        # cpdist = CPDist(
        #     BaseDistConfig(horizon=H, rank=R, embedding_dim=D, vocab_size=V)
        # )
        dims = [V] * H

        test_cases = []

        # Diagonal tensor
        diag_tensor = torch.zeros(dims)
        for i in range(min(dims)):
            idx = tuple([i] * H)
            diag_tensor[idx] = 1.0
        diag_tensor = diag_tensor / diag_tensor.sum()
        test_cases.append(diag_tensor)

        # Random tensor
        full = torch.randn(dims).sigmoid()
        full = full / full.sum()
        test_cases.append(full)

        for test_case in test_cases:
            t = tn.Tensor(test_case, ranks_cp=R, verbose=True)
            y = torch.tensor([[0, 1, 2]])

            p_tilde, p_tilde_scale_factors = select_margin_cp_tensor_batched(
                # (B, HR, V)
                cp_params=torch.stack(t.cores, dim=0).unsqueeze(0).reshape(B, H * R, V),
                ops=y,
            )
            z_tilde, z_tilde_scale_factors = select_margin_cp_tensor_batched(
                # (B, HR, V)
                cp_params=torch.stack(t.cores, dim=0).unsqueeze(0).reshape(B, H * R, V),
                ops=torch.full((B, H), -2, dtype=torch.long),
            )

            nll_actual = -torch.log(t[0, 1, 2]).reshape(1)
            torch.testing.assert_close(nll_hat, nll_actual)


if __name__ == "__main__":
    unittest.main()
