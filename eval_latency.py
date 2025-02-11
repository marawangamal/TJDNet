import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)

import time
from statistics import mean, stdev

from distributions._base import BaseDistConfig
from distributions.tpnet import TensorParamNetConfig
from helpers import get_test_samples
from models._tjd import TJDConfig
from models.tjdgpt2 import TJDGPT2
from models.tjdllama import TJDLLAMA

import line_profiler

GEN_KWARGS = {
    "max_new_tokens": 256,
    "num_beams": 1,
    "top_k": 200,
    "do_sample": False,
}


def benchmark_model(
    model, tokenizer, generate_fn, num_runs=10, num_warmup=3, device="cuda"
):
    input_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids]).to(device)

    print("Warming up...")
    for _ in tqdm(range(num_warmup), desc="Warmup", leave=False):
        _ = generate_fn(model, input_ids)

    latencies = []
    for i in tqdm(range(num_runs), desc="Benchmark", leave=False):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = generate_fn(model, input_ids)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start_time)

    return {
        "mean": mean(latencies),
        "std": stdev(latencies) if len(latencies) > 1 else 0,
        "min": min(latencies),
        "max": max(latencies),
        "all_latencies": latencies,
    }


@line_profiler.profile
def manual_gen_from_model(
    model,
    input_ids,
    max_new_tokens=100,
    num_beams=1,
    top_k=200,
    do_sample=False,
    base_transformer_attr_name="model",
):
    with torch.no_grad():
        for _ in range(max_new_tokens):
            transformer_base = getattr(model, base_transformer_attr_name)
            outputs = transformer_base(
                input_ids,
            )
            last_hidden_state = outputs.last_hidden_state
            next_token_logits = model.lm_head(last_hidden_state[:, -1, :])
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

            if do_sample:
                # Apply top-k filtering
                top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k)
                next_token = top_k_indices[0][torch.multinomial(top_k_probs[0], 1)]
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_probs, dim=-1).to(input_ids.device)

            # # Check for EOS token
            # if next_token.item() == model.config.eos_token_id:
            #     break

            # Append next token to input sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    return input_ids


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run the model on (cuda or cpu)",
    )
    args = parser.parse_args()

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    gp2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    device_map = "auto" if args.device == "cuda" else "cpu"

    GPT2_KWARGS = {
        "vocab_size": len(gp2_tokenizer),
        "n_layer": 6,
        "n_head": 6,
        "dropout": 0.1,
    }

    MODELS = {
        "gpt2-manual": {
            "model_fn": lambda: GPT2LMHeadModel(GPT2Config(**GPT2_KWARGS)).to(
                args.device
            ),
            "tokenizer": gp2_tokenizer,
            "generate_fn": lambda model, input_ids: manual_gen_from_model(
                model,
                input_ids,
                base_transformer_attr_name="transformer",
                **GEN_KWARGS,
            ),
        },
        "gpt2-tjd": {
            "model_fn": lambda: TJDGPT2(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=len(gp2_tokenizer),
                        horizon=1,
                        rank=1,
                        param_net=TensorParamNetConfig(),
                    ),
                    model_head="base",
                    model_kwargs=GPT2_KWARGS,
                )
            ).to(args.device),
            "tokenizer": gp2_tokenizer,
            "generate_fn": lambda model, input_ids: model.generate(
                input_ids,
                **GEN_KWARGS,
            ),
        },
        # "llama": {
        # "model_fn": lambda: AutoModelForCausalLM.from_pretrained(
        #         "meta-llama/Llama-2-7b-chat-hf",
        #         low_cpu_mem_usage=True,
        #         device_map="auto",
        #     ),
        #     "tokenizer": base_tokenizer,
        #     "generate_fn": lambda model, input_ids: model.generate(
        #         input_ids,
        #         **gen_kwargs,
        #     ),
        # },
        # "llama-manual": {
        #     "model_fn": lambda: AutoModelForCausalLM.from_pretrained(
        #         "meta-llama/Llama-2-7b-chat-hf",
        #         low_cpu_mem_usage=True,
        #         device_map=device_map,
        #     ),
        #     "tokenizer": base_tokenizer,
        #     "generate_fn": lambda model, input_ids: manual_gen_from_model(
        #         model,
        #         input_ids,
        #         **gen_kwargs,
        #     ),
        # },
        # "tjd-llama": {
        #     "model_fn": lambda: TJDLLAMA(
        #         TJDConfig(
        #             base_dist=BaseDistConfig(
        #                 vocab_size=len(base_tokenizer),
        #                 horizon=1,
        #                 rank=1,
        #                 param_net=TensorParamNetConfig(),
        #             ),
        #             model_head="base",
        #         )
        #     ).to(args.device),
        #     "tokenizer": base_tokenizer,
        #     "generate_fn": lambda model, input_ids: model.generate(
        #         input_ids,
        #         **gen_kwargs,
        #     ),
        # },
        # "tjd-llama-ts": {
        #     "model_fn": lambda: TJDLLAMA(
        #         TJDConfig(
        #             base_dist=BaseDistConfig(
        #                 vocab_size=len(base_tokenizer),
        #                 horizon=1,
        #                 rank=1,
        #                 param_net=TensorParamNetConfig(),
        #             ),
        #             model_head="base",
        #         )
        #     ).cuda(),
        #     "tokenizer": base_tokenizer,
        #     "generate_fn": lambda model, input_ids: get_test_samples(
        #         model=model,
        #         tokenizer=base_tokenizer,
        #         **gen_kwargs,
        #     ),
        # },
    }

    # Run benchmarks
    print(f"Starting benchmarks ({args.device})...")
    results = {}
    for model_name, config in MODELS.items():
        print(f"\nBenchmarking {model_name}...")
        try:
            results[model_name] = benchmark_model(
                config["model_fn"](),
                config["tokenizer"],
                config["generate_fn"],
                num_runs=1,
                num_warmup=0,
                device=args.device,
            )
            print(f"Results for {model_name}")
            print(
                f"Mean latency: {results[model_name]['mean']:.3f}s ± {results[model_name]['std']:.3f}s"
            )
            print(f"Min latency: {results[model_name]['min']:.3f}s")
            print(f"Max latency: {results[model_name]['max']:.3f}s")
        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            continue

    # Print results
    print("\nBenchmark Results:")
    print("-" * 50)
    # for model_name, stats in results.items():
    for model_name, stats in results.items():
        print(f"\n{model_name}:")
        print(f"Mean latency: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        print(f"Min: {stats['min']:.3f}s | Max: {stats['max']:.3f}s")
