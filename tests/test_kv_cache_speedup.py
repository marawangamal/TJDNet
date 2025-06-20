import time
import torch
from tjdnet.models.tjdhf import TJDHuggingFace
from tjdnet.models.tjd import TJDConfig
from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig
from transformers import AutoTokenizer

MODEL_NAME = "lmsys/vicuna-7b-v1.5"
horizon, rank, hidden_dim, max_new_tokens = 2, 2, 64, 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(
    "The quick brown fox jumps over the lazy dog.", return_tensors="pt"
).input_ids


def make_model(head):
    h = 1 if head == "base" else horizon
    return TJDHuggingFace(
        config=TJDConfig(
            model_head=head,
            model_head_config=BaseDistConfig(
                vocab_size=-1,
                horizon=h,
                rank=rank,
                param_net=TensorParamNetConfig(hidden_dim=hidden_dim),
            ),
        ),
        auto_model_kwargs={"pretrained_model_name_or_path": MODEL_NAME},
        train_mode="full",
        lora_rank=1,
    )


def bench(model, use_cache):
    torch.cuda.empty_cache()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    _ = model.generate(
        input_ids=input_ids.to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=1,
        use_cache=use_cache,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time() - t0


if __name__ == "__main__":
    results = []
    for head in ["base", "cp"]:
        model = make_model(head).to("cuda" if torch.cuda.is_available() else "cpu")
        for cache in [False, True]:
            t = bench(model, cache)
            results.append((head, "On" if cache else "Off", t))
    print(f"\nKV-Caching Generation Speed Benchmark ({MODEL_NAME}), horizon=2, rank=2)")
    print("+------------+------------+----------+----------+")
    print(
        "| Model Head | KV-caching | Time (s) | Speedup  |\n+------------+------------+----------+----------+"
    )
    for head in ["base", "cp"]:
        t_off = next(t for h, c, t in results if h == head and c == "Off")
        t_on = next(t for h, c, t in results if h == head and c == "On")
        print(f"| {head:10} | {'Off':10} | {t_off:8.4f} | {'-':8} |")
        print(f"| {head:10} | {'On':10} | {t_on:8.4f} | {t_off/t_on:8.2f}x |")
    print("+------------+------------+----------+----------+")
