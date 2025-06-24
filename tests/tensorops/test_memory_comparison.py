import unittest
import torch
import gc

from tjdnet.tensorops.cp import (
    select_margin_cp_tensor_batched,
    select_margin_cp_tensor_batched_w_decoder,
)


def decoder_method_select(B, D, R, H, V, device):
    """Original memory-intensive approach"""
    cp_params = torch.randn(B, R, H, D, device=device)
    ops = torch.randint(0, V, (B, H), device=device)
    decoder = torch.randn(D, V, device=device)
    select_margin_cp_tensor_batched_w_decoder(cp_params, ops, decoder)


def gather_method_select(B, D, R, H, V, device):
    """Original memory-intensive approach"""
    cp_params = torch.randn(B, R, H, V, device=device)
    ops = torch.randint(0, V, (B, H), device=device)
    select_margin_cp_tensor_batched(cp_params, ops)


def gather_method_margin(B, D, R, H, V, device):
    """Original memory-intensive approach"""
    cp_params = torch.randn(B, R, H, V, device=device)
    ops = torch.full((B, H), -2, dtype=torch.long, device=device)
    select_margin_cp_tensor_batched(cp_params, ops)


def decoder_method_margin(B, D, R, H, V, device):
    """Original memory-intensive approach"""
    cp_params = torch.randn(B, R, H, D, device=device)
    ops = torch.full((B, H), -2, dtype=torch.long, device=device)
    decoder = torch.randn(D, V, device=device)
    select_margin_cp_tensor_batched_w_decoder(cp_params, ops, decoder)


class TestMemoryComparison(unittest.TestCase):
    """Simple memory usage comparison between CP tensor functions."""

    def setUp(self):
        """Set up test environment."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        self.device = torch.device("cuda")

    def measure_memory(self, func, *args, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        result = func(*args, **kwargs)
        peak_memory = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        gc.collect()
        return result, peak_memory

    def test_memory_comparison(self):
        """Compare memory usage between the two functions with random values."""

        # hparams
        B, D, H, R, V = 32, 4096, 32, 32, 100000
        device = self.device
        methods = [
            (
                gather_method_select,
                "gather_method_select",
                (B, D, R, H, V, device),
            ),
            (
                decoder_method_select,
                "decoder_method_select",
                (B, D, R, H, V, device),
            ),
            (
                gather_method_margin,
                "gather_method_margin",
                (B, D, R, H, V, device),
            ),
            (
                decoder_method_margin,
                "decoder_method_margin",
                (B, D, R, H, V, device),
            ),
        ]
        results = {
            "gather_method_select": -1,
            "decoder_method_select": -1,
            "gather_method_margin": -1,
            "decoder_method_margin": -1,
        }
        for method, name, args in methods:
            result, mem = self.measure_memory(method, *args)
            print(f"{name}: {mem/1024**2:.2f} MB")
            results[name] = mem

        # Assert that decoder function uses less memory
        self.assertLess(
            results["decoder_method_select"],
            results["gather_method_select"],
            "Decoder function should use less memory",
        )

        self.assertLess(
            results["decoder_method_margin"],
            results["gather_method_margin"],
            "Decoder function should use less memory",
        )

    def tearDown(self):
        """Clean up after tests."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    unittest.main()
