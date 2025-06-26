import unittest
import torch
import gc

from tjdnet.distributions.cpe import CPEffDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.types import PositivityFuncType


# Memory efficient implementation
def cpe_method(B, D, H, R, V, device, positivity_func: PositivityFuncType = "safe_exp"):
    config = BaseDistConfig(
        vocab_size=V,
        horizon=H,
        rank=R,
        embedding_dim=D,
        positivity_func=positivity_func,
    )
    model = CPEffDist(config).to(device)
    x = torch.randn(B, D, device=device)
    y = torch.randint(0, V, (B, H), device=device)
    model(x, y)
    return model


def cp_method(B, D, H, R, V, device, positivity_func: PositivityFuncType = "safe_exp"):
    config = BaseDistConfig(
        vocab_size=V,
        horizon=H,
        rank=R,
        embedding_dim=D,
        positivity_func=positivity_func,
    )
    model = CPDist(config).to(device)
    x = torch.randn(B, D, device=device)
    y = torch.randint(0, V, (B, H), device=device)
    model(x, y)
    return model


class TestCPEffVsCPDistMemory(unittest.TestCase):
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
        # Explicitly delete model and collect
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return peak_memory, param_mem

    def test_memory_comparison(self):
        # hparams
        # B, D, H, R, V = 32, 4096, 32, 32, 30000
        B, D, H, R, V = 64, 1024, 32, 64, 30000
        device = self.device
        methods = [
            (cp_method, "CPDist+safe_exp", (B, D, H, R, V, device, "safe_exp")),
            (cpe_method, "CPEffDist+safe_exp", (B, D, H, R, V, device, "safe_exp")),
            (cp_method, "CPDist+sigmoid", (B, D, H, R, V, device, "sigmoid")),
            (cpe_method, "CPEffDist+sigmoid", (B, D, H, R, V, device, "sigmoid")),
            (cp_method, "CPDist+relu", (B, D, H, R, V, device, "relu")),
            (cpe_method, "CPEffDist+relu", (B, D, H, R, V, device, "relu")),
            (cp_method, "CPDist+leaky_relu", (B, D, H, R, V, device, "leaky_relu")),
            (cpe_method, "CPEffDist+leaky_relu", (B, D, H, R, V, device, "leaky_relu")),
        ]
        results = {"CPDist": -1, "CPEffDist": -1}
        param_percents = {"CPDist": -1.0, "CPEffDist": -1.0}
        for method, name, args in methods:
            peak_memory, param_mem = self.measure_memory(method, *args)
            percent = 100 * param_mem / peak_memory if peak_memory > 0 else 0
            print(
                f"{name}: {peak_memory/1024**2:.2f} MB (params: {param_mem/1024**2:.2f} MB, {percent:.2f}% of peak)"
            )
            results[name] = peak_memory
            param_percents[name] = percent
        # Assert that CPEffDist uses less or equal memory than CPDist
        self.assertLessEqual(
            results["CPEffDist"],
            results["CPDist"],
            "CPEffDist should use less or equal memory than CPDist",
        )

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    unittest.main()
