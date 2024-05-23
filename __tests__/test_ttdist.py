import unittest
import torch
from TJDNet import TTDist, sample_from_tensor_dist


def normalize_matrix(matrix):
    """Placeholder normalization function, replace with the actual implementation."""
    norm_factor = torch.norm(matrix, dim=0, keepdim=True)
    return matrix / norm_factor


class TestTTDist(unittest.TestCase):
    def setUp(self):
        # Setup for BTTN instance
        R, D, B, n_core_repititions = 4, 3, 1, 2  # Example dimensions
        alpha = torch.randn(B, R)
        beta = torch.randn(B, R)
        core = torch.randn(B, R, D, R)
        self.bttn = TTDist(alpha, beta, core, n_core_repititions)
        self.n_core_repititions = n_core_repititions
        self.vocab_size = D

    def test_beam_search_single_batch(self):
        """Test beam_search on a single batch."""
        n_beams = 5
        beams, beam_probs = self.bttn._beam_search(n_beams=n_beams)

        # Ensure that the method returns the correct number of beams
        self.assertEqual(len(beams), min(n_beams, self.vocab_size))
        self.assertEqual(len(beam_probs), min(n_beams, self.vocab_size))

        # Ensure the sequence lengths are correct for each beam
        for beam in beams:
            self.assertEqual(len(beam), self.n_core_repititions)

        # # Check the validity of returned beams
        # for beam in beams:
        #     self.assertTrue(all(isinstance(b, torch.Tensor) for b in beam))
        #     self.assertTrue(
        #         all(0 <= b.item() < D for b in beam)
        #     )  # D is the size of the core's dimension

        # # Check that beam probabilities are tensors and their values are plausible
        for prob in beam_probs:
            self.assertIsInstance(prob, torch.Tensor)
            self.assertTrue(prob > 0)

    def test_get_prob_and_norm_with_single_timestep(self):
        """Test get_prob_and_norm method."""

        batch_size, rank, vocab_size, output_size = 5, 1, 10, 1
        alpha, beta = torch.ones(batch_size, rank), torch.ones(batch_size, rank)
        core = torch.zeros(batch_size, rank, vocab_size, rank)
        target = (torch.ones(batch_size, output_size) * 5).long()
        for b in range(batch_size):
            core[b, :, 5, :] = b + 1

        ttdist = TTDist(alpha, beta, core, output_size)
        prob_tilde, norm_constant = ttdist.get_prob_and_norm(target)

        self.assertTrue(torch.equal(prob_tilde, torch.arange(1, batch_size + 1)))
        self.assertTrue(torch.equal(norm_constant, torch.arange(1, batch_size + 1)))

    # def test_get_prob_and_norm_operating_range(self):
    #     """Test get_prob_and_norm method."""
    #     # alpa, beta = ones
    #     # for std in np.linspace(0, 10, 100):
    #     #   core = randn(mu=0, std=std)
    #     #   assert(contraction.isFinite())
    #     raise NotImplementedError("Test not implemented yet.")

    def test_sample_from_true_dist(self):

        non_zero_indices_list = [
            [
                [0],
            ],
            [
                [0, 1, 0, 1],
                [0, 0, 1, 1],
            ],
        ]
        for non_zero_indices in non_zero_indices_list:

            non_zero_indices = torch.tensor(non_zero_indices)

            vocab_size = 4
            n_dims = non_zero_indices.size(1)
            n_samples = 10
            prob_tensor = torch.zeros(*[vocab_size for _ in range(n_dims)])
            prob_tensor[tuple(non_zero_indices.t())] = 1.0
            prob_tensor = prob_tensor / prob_tensor.sum()

            samples = sample_from_tensor_dist(prob_tensor, n_samples)
            for sample in samples:
                self.assertTrue(
                    any(
                        torch.equal(sample, valid_sample)
                        for valid_sample in non_zero_indices
                    )
                )

    def test_ttdist_leanrable(self):
        """Test get_prob_and_norm method."""
        n_iters = 1000
        batch_size = 5
        rank = 2
        vocab_size = 10
        output_size = 1

        true_dist = torch.abs(torch.randn(*[vocab_size for _ in range(output_size)]))
        true_dist = true_dist / true_dist.sum()  # P(d1, d2, ..., dN)

        # Sample `batch_size` random samples from the true distribution
        samples = sample_from_tensor_dist(true_dist, batch_size)

        alpha = torch.nn.Parameter(torch.randn(batch_size, rank))
        beta = torch.nn.Parameter(torch.randn(batch_size, rank))
        core = torch.nn.Parameter(torch.randn(batch_size, rank, vocab_size, rank))

        optimizer = torch.optim.Adam([alpha, beta, core], lr=1e-2)

        for i in range(n_iters):
            optimizer.zero_grad()

            # Forward pass:
            alpha_pos = torch.nn.functional.relu(alpha) + 1e-6
            beta_pos = torch.nn.functional.relu(beta) + 1e-6
            core_pos = torch.nn.functional.relu(core) + 1e-6
            ttdist = TTDist(alpha_pos, beta_pos, core_pos, output_size)
            probs_tilde, norm_constant = ttdist.get_prob_and_norm(samples)
            loss = (-torch.log(probs_tilde) + torch.log(norm_constant)).mean()

            # Backward pass:
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss.item()}")

        self.assertTrue(loss.item() < 1e-2)


if __name__ == "__main__":
    unittest.main()
