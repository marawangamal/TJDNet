from typing import Optional
import unittest

import torch

from tjdnet.utils import sample_topk, spec_sample_v2


class MockModel:
    def __init__(self, batch_size, horizon, vocab_size, embd_dim):
        self.batch_size = batch_size
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.encoder = torch.nn.Embedding(vocab_size, embd_dim)  # (V, D)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embd_dim, embd_dim),  # (D, D)
            torch.nn.ReLU(),
            torch.nn.Linear(embd_dim, vocab_size),  # (D, V)
        )
        self.x = torch.randint(0, vocab_size, (batch_size, 1))

    def prob_y_bar_x(self, y: Optional[torch.Tensor]) -> torch.Tensor:
        """Computes sequence likelihood P(yh|x, y1:h-1) for each h.

        Args:
            y (torch.Tensor): Input tensor of shape (B, H).
        Returns:
            torch.Tensor: Sequence likelihood. Shape (B, H, V).

        """
        # Mock probability distribution
        y_prime = torch.cat((self.x, y), dim=1) if y is not None else self.x  # (B, H)
        emb_y = self.encoder(y_prime)  # (B, H, D)
        pys = torch.nn.functional.softmax(self.decoder(emb_y), dim=-1)  # (B, H, V)
        return pys  # (B, H+1, V)

    def sample(self):
        """Samples tokens from P(yk|x, y1:k-1) for k in [1, H].

        Args:
            None
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of tensors (y_hat, qy).
             - y_hat (torch.Tensor): Draft tokens from model_q. Shape: (B, H).
             - qy (torch.Tensor): Draft probabilities from model_q. Shape: (B, H, V).

        """
        # Mock sampling
        y_out = None
        pys = []
        for h in range(self.horizon):
            py_bar_xy = self.prob_y_bar_x(y_out)  # (B, h'+1, V) h' in [1, H]
            pyh_bar_xy = py_bar_xy[:, -1, :]  # (B, V)
            pys.append(pyh_bar_xy)  # (B, V)
            # Sample from the model
            y = sample_topk(pyh_bar_xy, top_k=1)  # (B, 1)
            y_out = y if y_out is None else torch.cat((y_out, y), dim=1)  # (B, h')

        if y_out is None:
            raise ValueError("y_out is None, check the model sampling logic.")

        return y_out, torch.stack(pys, dim=1)  # (B, H), (B, H, V)


class TestSpecSample(unittest.TestCase):

    def test_single_level_grouping(self):

        tokens_proposed = 0
        tokens_accepted = 0

        for _ in range(1000):
            model = MockModel(batch_size=1, horizon=3, vocab_size=5, embd_dim=4)
            # Test grouping by only model_head
            y = spec_sample_v2(
                # model_p: y -> p(y1|x), p(y2|x, y1), ..., p(yh|x, y1:H-1). Shape: (B, H) -> (B, H, V)
                model_p=lambda y: model.prob_y_bar_x(y)[:, :-1, :],  # (B, H, V)
                # {} -> y_hat, q(y1|x), q(y2|x, y1), ..., q(yh|x, y1:H-1). Shape: None -> (B, H), (B, H, V)
                model_q=model.sample,
                sample_fn=lambda p: sample_topk(p, top_k=1).squeeze(-1),
            )  # (B', H') -- H' <= H_tgt if not all tokens are accepted

            tokens_proposed += model.horizon
            tokens_accepted += y.shape[1]

        # Expect acceptance rate > 0.5
        acceptance_rate = tokens_accepted / tokens_proposed
        self.assertEqual(
            acceptance_rate, 1.0, msg=f"Acceptance rate: {acceptance_rate:.2f} < 1.0"
        )


if __name__ == "__main__":
    unittest.main()
