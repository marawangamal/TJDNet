import unittest
import torch
from TJDNet import BTTN


def normalize_matrix(matrix):
    """Placeholder normalization function, replace with the actual implementation."""
    norm_factor = torch.norm(matrix, dim=0, keepdim=True)
    return matrix / norm_factor


class TestBTTNBeamSearch(unittest.TestCase):
    def setUp(self):
        # Setup for BTTN instance
        R, D, B, n_core_repititions = 4, 3, 1, 2  # Example dimensions
        alpha = torch.randn(B, R)
        beta = torch.randn(B, R)
        core = torch.randn(B, R, D, R)
        self.bttn = BTTN(alpha, beta, core, n_core_repititions)
        self.n_core_repititions = n_core_repititions
        self.vocab_size = D

    def test_beam_search_single_batch(self):
        """Test beam_search on a single batch."""
        n_beams = 5
        beams, beam_probs = self.bttn.beam_search(n_beams=n_beams)

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


if __name__ == "__main__":
    unittest.main()
