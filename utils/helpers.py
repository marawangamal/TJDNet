import os
import json
import random
import argparse
import subprocess

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from tjdnet.distributions import TJD_DISTS
from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig

from tjdnet.models.tjdhf import TJDConfig
from tjdnet.models.tjdhf import TJDHuggingFace


import uuid
import re

from tjdnet.models.tjdhf_v2 import TJDHuggingFaceV2
from tjdnet.models.tllama import TJDLlamaModel


def generate_wandb_id():
    """Generate a random 8-character ID with lowercase letters and numbers only."""
    # Generate UUID and remove hyphens
    raw_id = str(uuid.uuid4()).replace("-", "")
    # Keep only lowercase letters and numbers
    clean_id = re.sub(r"[^a-z0-9]", "", raw_id.lower())
    # Return first 8 characters
    return clean_id[:8]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on the ELI5 dataset.")

    # ------------------
    # Basic arguments
    # ------------------

    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for training."
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

    # ---------------
    # Model init args
    # ---------------

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Huggingface model id.",
    )
    parser.add_argument(
        "--model_head",
        type=str,
        default="cp",
        help="Type of factorization to use for the model.",
        choices=TJD_DISTS.keys(),
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden size of model head.",
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
        help="Horizon for TJD models. (E.g. if horizon=2 the model will make 2x less forward passes)",
    )
    parser.add_argument(
        "--positivity_func",
        type=str,
        default="exp",
        choices=["sq", "abs", "exp"],
        help="Positivity function to use for MPSDist.",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="random",
        choices=[
            "random",  # Completely random initialization
            "pretrained",  # Initialize the model tensor head with pretrained weights
        ],
        help="Initialization method for model head - pretrained or random",
    )
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="draft",
        choices=["joint", "draft"],
        help="Loss mode for training.",
    )
    parser.add_argument(
        "--joint_loss_lambda",
        type=float,
        default=1,
        help="Weight for target model loss in joint loss.",
    )
    parser.add_argument(
        "--train_mode",
        default="lora",
        choices=[
            "full",
            "lora",
        ],
        help="Training mode for the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="Rank of the tensor train decomposition for LORA training.",
    )
    parser.add_argument(
        "--use_memory_efficient_loss",
        default=False,
        action="store_true",
        help="Whether to use a memory efficient loss function.",
    )

    # ------------------
    # Data arguments
    # ------------------

    parser.add_argument(
        "--dataset",
        type=str,
        default="stemp",
        help="Type of dataset to use for training.",
        choices=[
            "shakespeare",
            "wikitext",
            "sharegpt",
            "gsm8k",
            "stemp",
            "snum",
            "sbase",
        ],
    )

    # ------------------
    # Eval arguments (EXCLUDED FROM EXP NAME)
    # ------------------
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate during evaluation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Retain only the top_k most likely tokens, clamp others to have 0 probability",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gen_mode",
        type=str,
        default="draft",
        choices=["base", "draft", "speculative"],
    )
    parser.add_argument(
        "--compute_acc", action="store_true", help="Whether to compute accuracy"
    )

    # ------------------
    # Trainer arguments (EXCLUDED FROM EXP NAME)
    # ------------------

    parser.add_argument(
        "--logging_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch"],
        help="Logging strategy for the trainer.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Logging frequency for the trainer.",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch"],
        help="Evaluation strategy for the trainer.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1,
        help="Evaluation frequency for the trainer.",
    )
    parser.add_argument(
        "--generate_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch", "no"],
        help="Strategy for text generation during training",
    )
    parser.add_argument(
        "--generate_steps",
        type=int,
        default=1000,
        help="Number of steps between generations if strategy is 'steps'",
    )
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=10000,
        help="Maximum number of samples to load from the dataset.",
    )

    # ---
    # MISC (EXCLUDED FROM EXP NAME)
    # ---
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Whether to disable wandb logging.",
        default=False,
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tjdnet",
        help="Wandb project prefix name.",
    )

    args = parser.parse_args()
    validate_args(args)
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
        commit_message = (
            subprocess.check_output(["git", "log", "-1", "--pretty=%s"])
            .decode("ascii")
            .strip()
        )
        # Add this line to get the current branch name
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )
        return {
            "commit_hash": commit_hash,
            "commit_message": commit_message,
            "branch": branch,
        }
    except:
        return {
            "commit_hash": "Git commit hash not available",
            "commit_message": "Git commit message not available",
            "branch": "unknown",
        }


def validate_args(args):
    rules = [
        # {
        #     "message": "Model does not support batch_size > 1",
        #     "condition": lambda: not (
        #         args.model in ["gpt2"] and args.acc_batch_size > 1
        #     ),
        # }
    ]

    for rule in rules:
        if not rule["condition"]():
            raise ValueError(rule["message"])


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


def get_auto_tokenizer(model_name):
    """Get the tokenizer for a given model name."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        raise

    # Set padding token
    tokenizer.pad_token = "$"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"
    return tokenizer


def get_model_and_tokenizer(args):
    tokenizer = get_auto_tokenizer(args.model)
    model_config = TJDConfig(
        model_head=args.model_head,
        model_head_config=BaseDistConfig(
            vocab_size=len(tokenizer),
            horizon=args.horizon,
            rank=args.rank,
            param_net=TensorParamNetConfig(
                hidden_dim=args.hidden_dim,
                positivity_func=args.positivity_func,
            ),
        ),
        init_method=args.init_method,
        loss_mode=args.loss_mode,
    )
    model = TJDHuggingFace(
        model_config,
        auto_model_kwargs=dict(
            pretrained_model_name_or_path=args.model,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ),
        train_mode=args.train_mode,
        lora_rank=args.lora_rank,
    )
    return model, tokenizer


def get_model_and_tokenizer_tjdhfv2(args):
    tokenizer = get_auto_tokenizer(args.model)
    model = TJDHuggingFaceV2(
        auto_model_kwargs=dict(
            pretrained_model_name_or_path=args.model,
            low_cpu_mem_usage=True,
        ),
    )
    return model, tokenizer


def get_model_and_tokenizer_tjdllama(args):
    tokenizer = get_auto_tokenizer(args.model)
    model = TJDLlamaModel.from_pretrained(
        args.model,
    )
    return model, tokenizer


def get_model_and_tokenizer_nowrap(args):
    tokenizer = get_auto_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
    )
    return model, tokenizer


# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', low_cpu_mem_usage=True)