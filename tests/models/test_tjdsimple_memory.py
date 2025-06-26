import unittest
import torch
import gc
from tjdnet.models.tjdsimple import TJDSimple, TJDSimpleConfig


def standard_backward_method(B, H, V, device, config):
    model = TJDSimple(config).to(device)
    x = torch.randint(0, V, (B, H), device=device)
    y = torch.randint(0, V, (B, H), device=device)
    out = model(input_ids=x, labels=y)
    out["loss"].mean().backward()
    del model, x, y, out


def memory_efficient_backward_method(B, H, V, device, config):
    model = TJDSimple(config).to(device)
    x = torch.randint(0, V, (B, H), device=device)
    y = torch.randint(0, V, (B, H), device=device)
    if hasattr(model, "forward_backward"):
        out = model.forward_backward(input_ids=x, labels=y)
        out["loss"].mean()
    else:
        raise RuntimeError("Model does not have forward_backward method")
    del model, x, y, out


def run_memory_benchmark(methods):
    results = {}
    for name, func, args in methods:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        func(*args)
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"{name}: {peak_memory/1024**2:.2f} MB")
        results[name] = peak_memory
        torch.cuda.empty_cache()
        gc.collect()
    return results


class TestTJDSimpleMemoryMinimal(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")
        self.device = torch.device("cuda")

    def test_backward_memory_comparison(self):
        B, H, V = 32 * 128, 8, 50257
        device = self.device
        config = TJDSimpleConfig(
            model_name="gpt2",
            model_head="multihead",
            horizon=H,
            train_mode="full",
            lora_rank=32,
            positivity_func="safe_exp",
        )
        methods = [
            (
                "MemoryEfficientBackward",
                memory_efficient_backward_method,
                (B, H, V, device, config),
            ),
            ("StandardBackward", standard_backward_method, (B, H, V, device, config)),
        ]
        results = run_memory_benchmark(methods)
        self.assertLessEqual(
            results["MemoryEfficientBackward"],
            results["StandardBackward"],
            "MemoryEfficientBackward should use less or equal memory than StandardBackward",
        )


if __name__ == "__main__":
    unittest.main()
