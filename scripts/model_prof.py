import argparse
import torch
import torch.autograd.profiler as profiler

from models.tjdgpt2.tjdgpt2 import TJDGPT2
from models.tjdgpt2.tjdgpt2 import TJDGPT2


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on the ELI5 dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="cp",
        help="Type of model to use (gpt2 or tgpt2).",
        choices=[
            "cp",
            "mps",
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    # Configuration for a smaller GPT2 model (baby GPT)
    # kwargs = parse_args()
    seq_len = 32
    kwargs = parse_args()
    model_config = {
        "model": "cp",
        "vocab_size": 128,
        "n_embd": 64,
        "n_layer": 2,
        "n_head": 2,
        "dropout": 0.1,
        "rank": 2,
        "horizon": 2,
        **kwargs,
    }
    model = TJDGPT2(**model_config)

    # warm-up
    inp = torch.randint(0, model_config["vocab_size"], (1, seq_len))
    labels = torch.randint(0, model_config["vocab_size"], (1, seq_len))
    model(input_ids=inp, labels=labels)

    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        loss = model(inp, labels)

print(
    prof.key_averages(group_by_stack_n=10).table(
        sort_by="self_cpu_time_total", row_limit=20
    )
)
