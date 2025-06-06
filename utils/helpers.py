import os
import json
import random
import subprocess

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig

from tjdnet.models.tjdhf import TJDConfig
from tjdnet.models.tjdhf import TJDHuggingFace


import uuid
import re


def generate_wandb_id():
    """Generate a random 8-character ID with lowercase letters and numbers only."""
    # Generate UUID and remove hyphens
    raw_id = str(uuid.uuid4()).replace("-", "")
    # Keep only lowercase letters and numbers
    clean_id = re.sub(r"[^a-z0-9]", "", raw_id.lower())
    # Return first 8 characters
    return clean_id[:8]


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


def get_model_and_tokenizer_nowrap(args):
    tokenizer = get_auto_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
    )
    return model, tokenizer
