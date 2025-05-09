import unittest

import torch

from tjdnet.utils import sample_topk, spec_sample_v2


class MockModel:
    def __init__(self, batch_size, horizon, vocab_size, embd_dim):
        self.batch_size = batch_size
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.embd = torch.nn.Embedding(vocab_size, embd_dim)  # (V, D)
        self.w = torch.nn.Linear(embd_dim, vocab_size)  # (D, V)
        self.x = torch.randint(0, vocab_size, (batch_size, horizon))

    def prob(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Input tensor of shape (B, H).
        Returns:
            torch.Tensor: Probability distribution of shape (B, H, V).
        """
        # Mock probability distribution
        x = self.embd(y)  # (B, H, D)
        pys = torch.nn.functional.softmax(self.w(x), dim=-1)  # (B, H, V)
        return pys

    def sample(self):
        """
        Args:
            p (torch.Tensor): Input tensor of shape (B, H, V).
        Returns:
            torch.Tensor: Sampled tensor of shape (B, H).
        """
        # Mock sampling
        x = self.embd(self.x)  # (B, H, D)
        pys = torch.nn.functional.softmax(self.w(x), dim=-1)  # (B, H, V)
        ys = torch.multinomial(pys.view(-1, self.vocab_size), 1).view(
            self.batch_size, self.horizon
        )
        return ys, pys


class TestSpecSample(unittest.TestCase):

    def test_single_level_grouping(self):

        tokens_proposed = 0
        tokens_accepted = 0

        for _ in range(1000):
            model = MockModel(batch_size=1, horizon=3, vocab_size=5, embd_dim=4)
            # Test grouping by only model_head
            y = spec_sample_v2(
                # model_p: y -> p(y1|x), p(y2|x, y1), ..., p(yh|x, y1:H-1). Shape: (B, H) -> (B, H, V)
                model_p=model.prob,
                # {} -> y_hat, q(y1|x), q(y2|x, y1), ..., q(yh|x, y1:H-1). Shape: None -> (B, H), (B, H, V)
                model_q=model.sample,
                sample_fn=lambda p: sample_topk(p, top_k=1).squeeze(-1),
            )  # (B', H') -- H' <= H_tgt if not all tokens are accepted

            tokens_proposed += model.horizon
            tokens_accepted += y.shape[1]

        # Expect acceptance rate > 0.5
        acceptance_rate = tokens_accepted / tokens_proposed
        print(f"Acceptance rate: {acceptance_rate:.2f}")
        self.assertGreater(acceptance_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
