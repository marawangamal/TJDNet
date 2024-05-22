import unittest
import torch
from TJDNet import TTDist


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

    # def test_ttdist_leanrable(self):
    #     """Test get_prob_and_norm method."""
    #     # alpa, beta = ones
    #     # core = randn(mu=0, std=std) // BxRxDxR
    #     # gt_idx = torch.randint(0, D, (B, T))
    #     # for i in range(100):
    #     #   probs = ttdist(alpha, beta, core, gt_idx)
    #     #   loss = -torch.log(probs).mean()
    #     #   loss.backward()
    #     #   optimizer.step()
    #     #  assert(loss.item() < 0.1)
    #     raise NotImplementedError("Test not implemented yet.")


if __name__ == "__main__":
    unittest.main()
