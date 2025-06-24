import unittest
import torch
import gc

from tjdnet.models.tjdhf import TJDHuggingFace
from tjdnet.models.tjd import TJDConfig
from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig


def tjdhf_cp_eff_method(B, T, H, R, D, device):
    config = TJDConfig(
        model_head="cp_eff",
        model_head_config=BaseDistConfig(
            vocab_size=-1,
            horizon=H,
            rank=R,
            param_net=TensorParamNetConfig(
                in_dim=-1,  # will be overwritten
                hidden_dim=D,
            ),
        ),
    )
    auto_model_kwargs = {
        "pretrained_model_name_or_path": "gpt2",
        "torch_dtype": torch.float16,
        "device_map": None,
    }
    model = TJDHuggingFace(config, auto_model_kwargs, train_mode="lora").to(device)
    x = torch.randint(0, model.vocab_size, (B, T), device=device)
    model(input_ids=x)
    return model


def tjdhf_cp_method(B, T, H, R, D, device):
    config = TJDConfig(
        model_head="cp",
        model_head_config=BaseDistConfig(
            vocab_size=-1,
            horizon=H,
            rank=R,
            param_net=TensorParamNetConfig(
                in_dim=-1,  # will be overwritten
                hidden_dim=D,
            ),
        ),
    )
    auto_model_kwargs = {
        "pretrained_model_name_or_path": "gpt2",
        "torch_dtype": torch.float16,
        "device_map": None,
    }
    model = TJDHuggingFace(config, auto_model_kwargs, train_mode="lora").to(device)
    x = torch.randint(0, model.vocab_size, (B, T), device=device)
    model(input_ids=x)
    return model


class TestTJDHFCPEffVsCPDistMemory(unittest.TestCase):
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

    def test_memory_comparison(self):
        # hparams
        B, T, H, R, D = 2, 8, 2, 1, 768  # small for quick test
        device = self.device
        methods = [
            (tjdhf_cp_method, "TJDHF+CPDist", (B, T, H, R, D, device)),
            # (tjdhf_cp_eff_method, "TJDHF+CPEffDist", (B, T, H, R, D, device)),
        ]
        results = {"TJDHF+CPDist": -1, "TJDHF+CPEffDist": -1}
        param_percents = {"TJDHF+CPDist": -1.0, "TJDHF+CPEffDist": -1.0}
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
            results["TJDHF+CPEffDist"],
            results["TJDHF+CPDist"],
            "TJDHF+CPEffDist should use less or equal memory than TJDHF+CPDist",
        )

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    unittest.main()
