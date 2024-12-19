import json
import os
import argparse
import subprocess


import numpy as np
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.tjdgpt2 import TJDGPT2

from transformers import AutoTokenizer
from models.tjdgpt2 import TJDGPT2
from ctokenizers.char_tokenizer import CharTokenizer
from models.tjdllama import TJDLLAMA


# TODO: change horizon, horizon_eval to train_horizon, eval_horizon and eval_horizon should default to train_horizon if not specified
# TODO: put model arch in TJDGPT2 not in args
# TODO: add init method to args
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on the ELI5 dataset.")
    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--grad_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value for training.",
    )
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        help="Type of base model to use",
        choices=["gpt2", "llama"],
    )
    parser.add_argument(
        "--model_head",
        type=str,
        default="mps",
        help="Type of factorization to use for the model.",
        choices=[
            "cp",
            "mps",
            "umps",
            "full",
            "base",
        ],
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="word",
        help="Type of tokenizer to use for processing text.",
        choices=["char", "word"],
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Block size for model input sequences.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--positivity_func",
        type=str,
        default="exp",
        choices=["sq", "abs", "exp"],
        help="Positivity function to use for MPSDist.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=2,
        help="Rank of the tensor train decomposition.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="pretrained",
        choices=["pretrained", "p", "random", "r"],
        help="Initialization method for model head - pretrained (p) or random (r)",
    )
    parser.add_argument(
        "--freeze_base_model",
        default=False,
        action="store_true",
        help="Whether to freeze the base model during training.",
    )
    # Evaluation arguments
    parser.add_argument(
        "--horizon_eval",
        type=int,
        default=2,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate during evaluation.",
    )
    # Data Arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        help="Type of dataset to use for training.",
        choices=[
            "shakespeare",
            "wikitext",
            "sharegpt",
        ],
    )
    # Misc arguments
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_info():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        # Get just the first line of the commit message
        commit_message = (
            subprocess.check_output(["git", "log", "-1", "--pretty=%s"])
            .decode("ascii")
            .strip()
        )
        return {"commit_hash": commit_hash, "commit_message": commit_message}
    except:
        return {
            "commit_hash": "Git commit hash not available",
            "commit_message": "Git commit message not available",
        }


# TODO: add eval_horizon
def get_test_samples(
    model,
    tokenizer,
    prompt="\n",
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,
    num_beams=1,
    num_samples=1,
    print_output=False,
):
    # Inference
    model.eval()
    samples = []
    for i in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
        )
        sample = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if num_samples == 1:
            samples.append(sample)
        else:
            samples.append(f"[{i+1}] {sample}")

    if print_output:
        print("\n---\n".join(samples) + "\n")
    return "\n".join(samples)


def save_args(args, ckpt_dir):
    # Save args
    args_path = os.path.join(ckpt_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)


def load_args(ckpt_dir):
    args_path = os.path.join(ckpt_dir, "args.json")
    with open(args_path, "r") as f:
        args = json.load(f)
    return args


def get_tokenizer(args):
    pass


def get_model_and_tokenizer(args):
    # Tokenizer
    if args.model_type == "gpt2":
        tokenizer = (
            AutoTokenizer.from_pretrained("gpt2")
            if args.tokenizer_type == "word"
            else CharTokenizer(args.seq_len)
        )
    else:  # llama
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    # TODO: avoid this
    if args.tokenizer_type == "word" or args.model_type == "llama":
        tokenizer.pad_token = tokenizer.eos_token

    # Model configuration
    model_config = {
        "model_head": args.model_head,
        "vocab_size": (
            len(tokenizer.get_vocab())
            if hasattr(tokenizer, "get_vocab")
            else len(tokenizer)
        ),
        "dropout": args.dropout,
        "rank": args.rank,
        "horizon": args.horizon,
        "init_method": args.init_method,
        "freeze_base_model": args.freeze_base_model,
        "positivity_func": args.positivity_func,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Add LLaMA specific config
    if args.model_type == "llama":
        # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = TJDLLAMA(**model_config)
    else:
        model = TJDGPT2(**model_config)

    return model, tokenizer
