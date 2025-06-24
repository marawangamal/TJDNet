import unittest
import torch
import gc

from tjdnet.tensorops.cp import (
    select_margin_cp_tensor_batched,
    select_margin_cp_tensor_batched_w_decoder,
)


def decoder_method(B, d, R, T, D, device):
    """Original memory-intensive approach"""
    cp_params = torch.randn(B, R, T, d, device=device)
    ops = torch.randint(-2, D, (B, T), device=device)
    decoder = torch.randn(d, D, device=device)
    select_margin_cp_tensor_batched_w_decoder(cp_params, ops, decoder)


def gather_method(B, d, R, T, D, device):
    """Original memory-intensive approach"""
    cp_params = torch.randn(B, R, T, D, device=device)
    ops = torch.randint(-2, D, (B, T), device=device)
    select_margin_cp_tensor_batched(cp_params, ops)


class TestMemoryComparison(unittest.TestCase):
    """Simple memory usage comparison between CP tensor functions."""

    def setUp(self):
        """Set up test environment."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        self.device = torch.device("cuda")

    def measure_memory(self, func, *args, **kwargs):
        """Measure peak memory usage during function execution."""
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated()
        result = func(*args, **kwargs)
        peak_memory = torch.cuda.max_memory_allocated()

        torch.cuda.empty_cache()
        gc.collect()

        return result, peak_memory - initial_memory

    def test_memory_comparison(self):
        """Compare memory usage between the two functions with random values."""

        # hparams
        B, d, R, T, D = 32, 4096, 8, 128, 60000
        device = self.device
        methods = [
            (
                gather_method,
                "gather_method",
                (B, d, R, T, D, device),
            ),
            (
                decoder_method,
                "decoder_method",
                (B, d, R, T, D, device),
            ),
        ]
        results = {"decoder_method": -1, "gather_method": -1}
        for method, name, args in methods:
            result, mem = self.measure_memory(method, *args)
            print(f"{name}: {mem/1024**2:.2f} MB")
            results[name] = mem

        # Assert that decoder function uses less memory
        self.assertLess(
            results["decoder_method"],
            results["gather_method"],
            "Decoder function should use less memory",
        )

    def tearDown(self):
        """Clean up after tests."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    unittest.main()
