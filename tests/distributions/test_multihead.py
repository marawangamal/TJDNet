import unittest
import torch
import gc

from tjdnet.distributions.multihead import MultiHeadDist
from tjdnet.distributions._tjdist import BaseDistConfig


def multihead_method(B, D, H, V, device):
    config = BaseDistConfig(
        vocab_size=V,
        horizon=H,
        rank=1,
        embedding_dim=D,
        positivity_func="safe_exp",  # Not used, but required by config
    )
    model = MultiHeadDist(config).to(device)
    x = torch.randn(B, D, device=device)
    y = torch.randint(0, V, (B, H), device=device)
    model(x, y)
    return model


class TestMultiHeadMemory(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")
        self.device = torch.device("cuda")

    def measure_memory(self, func, *args):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        model = func(*args)
        peak_memory = torch.cuda.max_memory_allocated()
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return peak_memory, param_mem

    def measure_backward_memory(self, func, *args):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        model = func(*args)
        x = torch.randn(args[0], args[1], device=args[4])
        y = torch.randint(0, args[3], (args[0], args[2]), device=args[4])
        out = model(x, y)
        loss = out.mean() if hasattr(out, "mean") else out
        loss.backward()
        peak_memory = torch.cuda.max_memory_allocated()
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        del model, x, y, out, loss
        torch.cuda.empty_cache()
        gc.collect()
        return peak_memory, param_mem

    def test_memory_comparison(self):
        B, D, H, V = 32 * 128, 768, 2, 30000
        device = self.device
        methods = [
            (multihead_method, "MultiHeadDist+safe_exp", (B, D, H, V, device)),
            (multihead_method, "MultiHeadDist+sigmoid", (B, D, H, V, device)),
        ]
        for method, name, args in methods:
            peak_memory, param_mem = self.measure_memory(method, *args)
            percent = 100 * param_mem / peak_memory if peak_memory > 0 else 0
            print(
                f"{name}: {peak_memory/1024**2:.2f} MB (params: {param_mem/1024**2:.2f} MB, {percent:.2f}% of peak)"
            )

    def test_backward_memory_comparison(self):
        B, D, H, V = 32 * 128, 768, 2, 30000
        device = self.device
        methods = [
            (multihead_method, "MultiHeadDist+safe_exp", (B, D, H, V, device)),
            (multihead_method, "MultiHeadDist+sigmoid", (B, D, H, V, device)),
        ]
        for method, name, args in methods:
            peak_memory, param_mem = self.measure_backward_memory(method, *args)
            percent = 100 * param_mem / peak_memory if peak_memory > 0 else 0
            print(
                f"[BACKWARD] {name}: {peak_memory/1024**2:.2f} MB (params: {param_mem/1024**2:.2f} MB, {percent:.2f}% of peak)"
            )

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    unittest.main()
