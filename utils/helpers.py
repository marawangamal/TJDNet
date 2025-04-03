import os
import json
import random
import argparse
import subprocess

import torch
import numpy as np
from transformers import AutoTokenizer

from dataloaders.gsm8k import ChatTemplateGSM8k
from dataloaders.shakespeare import ChatTemplateShakespeare
from dataloaders.sharegpt import ChatTemplateShareGPT
from dataloaders.syn_number_bases import ChatTemplateSynNumBase
from dataloaders.syn_numbers import ChatTemplateSynNum
from dataloaders.syn_temp import ChatTemplateSynTemp
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.models._tjd import TJDConfig
from tjdnet.models.gpt2 import GPT2
from tjdnet.models.llama import LLAMA
from tjdnet.models.tjdgpt2 import TJDGPT2
from tjdnet.models.tjdllama import TJDLLAMA


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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on the ELI5 dataset.")

    # ------------------
    # Training arguments
    # ------------------

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
        "--seq_len",
        type=int,
        default=128,
        help="Block size for model input sequences.",
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

    # ---------------
    # Model arguments
    # ---------------

    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        help="Type of base model to use",
        choices=["gpt2", "llama7b", "llama13b", "llama70b", "gpt2r", "llamar"],
    )
    parser.add_argument(
        "--model_head",
        type=str,
        default="mps",
        help="Type of factorization to use for the model.",
        choices=[
            "cp",
            "ccp",
            "ucp",
            "mps",
            "umps",
            "full",
            "base",
        ],
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden size of model head.",
    )

    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of layers in the model head."
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function to use in the model head.",
    )
    parser.add_argument(
        "--use_layer_norm",
        default=False,
        action="store_true",
        help="Whether to use layer normalization in the model head.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Rank of the tensor train decomposition.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Horizon for TJD models. (E.g. if horizon=2 the model will make 2x less forward passes)",
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate.")
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
    # TODO: add to cli
    parser.add_argument(
        "--use_attn_layer",
        default=False,
        action="store_true",
        help="Whether to use attn layer in the model head.",
    )

    # Training mode
    parser.add_argument(
        "--train_mode",
        default="lora",
        choices=[
            "full",
            "last",
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
    # Evaluation arguments
    parser.add_argument(
        "--horizon_eval",
        type=int,
        default=1,
        help="Horizon for TJD models during evaluation. (Note: horizon_eval cannot be greater than horizon for some TJD dists)",
    )
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
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams to use during evaluation.",
    )
    parser.add_argument(
        "--gen_version", type=int, default=3, help="Generation method version"
    )
    # Data Arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
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
    # Tokenizer arguments
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="word",
        help="Type of tokenizer to use for processing text.",
        choices=["char", "word"],
    )
    # Misc arguments
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # ------------------
    # Trainer arguments
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
        default=68000,
        help="Maximum number of samples to load from the dataset.",
    )

    # ---
    # MISC
    # ---

    parser.add_argument(
        "--eval_only", action="store_true", help="Whether to only evaluate the model"
    )
    parser.add_argument(
        "--compute_acc", action="store_true", help="Whether to compute accuracy"
    )
    parser.add_argument(
        "--acc_batch_size",
        type=int,
        default=1,
        # GPT2 does not support batch_size > 1
        help="Batch size for computing accuracy. (NOTE: only models that support attention_mask can use batch_size > 1)",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="Wandb ID for resuming runs",
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
        {
            "message": "Model does not support batch_size > 1",
            "condition": lambda: not (
                args.model_type in ["gpt2"] and args.acc_batch_size > 1
            ),
        }
    ]

    for rule in rules:
        if not rule["condition"]():
            raise ValueError(rule["message"])


def get_test_samples(
    model,
    tokenizer,
    prompt="\n",
    max_new_tokens=128,
    top_k=50,
    do_sample=True,
    num_beams=1,
    num_samples=1,
    print_output=False,
    horizon=1,
    debug=True,
):
    # Inference
    model.eval()
    samples = []
    for i in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,
            horizon=horizon,
            early_stopping=True,  # For hf models this causes stopping when end token is reached (gpt2r)
            stop_token=tokenizer.eos_token_id,  # For tjd models this causes stopping when end token is reached
            stop_strings=[
                tokenizer.eos_token
            ],  # For tjd models this causes stopping when end token is reached
        )

        sample = tokenizer.decode(outputs[0])
        output_tokens = outputs[0]

        # Truncate the prompt from the output
        if sample.startswith(prompt):
            output_tokens = output_tokens[len(inputs[0]) :]
            sample = sample[len(prompt) :]

        if num_samples == 1:
            samples.append(sample)
        else:
            samples.append(f"[{i+1}] {sample}")

        # if debug:
        #     status = "Fail" if len(output_tokens) == max_new_tokens else "Pass"
        #     print(f"[{status}] Num tokens: {len(output_tokens)}")

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


def get_model_and_tokenizer(args):
    hf_model_name = {
        "llama7b": "meta-llama/Llama-2-7b-chat-hf",
        "llama13b": "meta-llama/Llama-2-13b-chat-hf",
        "llama70b": "meta-llama/Llama-2-70b-chat-hf",
        "gpt2": "gpt2",
    }[args.model_type]

    if args.tokenizer_type == "word":
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        # Note: cant simply add pad token -- unless we retrain a model embedding layer
        tokenizer.pad_token = "$"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = "left"
    else:
        raise NotImplementedError("CharTokenizer removed for now.")

    model_config = TJDConfig(
        base_dist=BaseDistConfig(
            vocab_size=len(tokenizer),
            horizon=args.horizon,
            rank=args.rank,
            param_net=TensorParamNetConfig(
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                activation=args.activation,
                use_layer_norm=args.use_layer_norm,
                positivity_func=args.positivity_func,
            ),
        ),
        model_head=args.model_head,
        init_method=args.init_method,
        train_mode=args.train_mode,
        lora_rank=args.lora_rank,
        use_memory_efficient_loss=args.use_memory_efficient_loss,
        use_attn_layer=(
            args.use_attn_layer if hasattr(args, "use_attn_layer") else False
        ),  # Backward compatibility
        model_kwargs={"hf_model_name": hf_model_name},
        # TODO: remove gen_version
        gen_version=(
            args.gen_version if hasattr(args, "gen_version") else 2
        ),  # Backward compatibility
    )

    model = {
        "llama7b": TJDLLAMA,
        "llama13b": TJDLLAMA,
        "llama70b": TJDLLAMA,
        "gpt2": TJDGPT2,
        "gpt2r": GPT2,
        "llamar": LLAMA,
    }[args.model_type](model_config)

    return model, tokenizer


def get_chat_template(args):
    chat_templates = {
        "gsm8k": ChatTemplateGSM8k,
        "shakespeare": ChatTemplateShakespeare,
        "sharegpt": ChatTemplateShareGPT,
        "snum": ChatTemplateSynNum,
        "sbase": ChatTemplateSynNumBase,
        "stemp": ChatTemplateSynTemp,
    }
    return chat_templates[args.dataset]


# # Tokenizer
# if args.model_type.startswith("gpt2"):
#     tokenizer = (
#         AutoTokenizer.from_pretrained("gpt2")
#         if args.tokenizer_type == "word"
#         else CharTokenizer(args.seq_len)
#     )

#     if args.tokenizer_type == "word":
#         tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
