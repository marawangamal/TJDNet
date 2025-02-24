import os
import json
import random
import argparse
import subprocess

import torch
import numpy as np
from transformers import AutoTokenizer

from ctokenizers.char_tokenizer import CharTokenizer
from data.gsm8k import ChatTemplateGSM8k
from data.shakespeare import ChatTemplateShakespeare
from data.sharegpt import ChatTemplateShareGPT
from data.sharegptv2 import ChatTemplateShareGPTV2
from data.syn_number_bases import ChatTemplateSynNumBase
from data.syn_numbers import ChatTemplateSynNum
from data.syn_temp import ChatTemplateSynTemp
from distributions._base import BaseDistConfig
from distributions.tpnet import TensorParamNetConfig
from models._tjd import TJDConfig
from models.gpt2 import GPT2
from models.llama import LLAMA
from models.tjdgpt2 import TJDGPT2
from models.tjdllama import TJDLLAMA


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
        default=256,
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
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        action="store_true",
        help="Whether to use a memory efficient loss function.",
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
        "--num_layers", type=int, default=2, help="Number of layers in the model head."
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
        default=2,
        help="Rank of the tensor train decomposition.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
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
        "--init_method",
        type=str,
        default="random",
        choices=[
            "random",  # Completely random initialization
            "pretrained",  # Initialize the model tensor head with pretrained weights
        ],
        help="Initialization method for model head - pretrained (p) or random (r)",
    )
    # Training mode
    parser.add_argument(
        "--train_mode",
        default="full",
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
        default=8,
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
        default=2,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate during evaluation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Retain only the top_k most likely tokens, clamp others to have 0 probability",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams to use during evaluation.",
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
        default="steps",
        choices=["steps", "epoch"],
        help="Logging strategy for the trainer.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Logging frequency for the trainer.",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch"],
        help="Evaluation strategy for the trainer.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Evaluation frequency for the trainer.",
    )
    parser.add_argument(
        "--generate_strategy",
        type=str,
        default="steps",
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
        "--wandb_id", type=str, default=None, help="Wandb ID for resuming runs"
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


# TODO: add eval_horizon
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


def get_tokenizer(args):
    pass


MODEL_NAME_MAP = {
    "llama7b": {"model_name": "meta-llama/Llama-2-7b-chat-hf"},
    "llama13b": {"model_name": "meta-llama/Llama-2-13b-chat-hf"},
    "llama70b": {"model_name": "meta-llama/Llama-2-70b-chat-hf"},
    "gpt2": {"model_name": "gpt2"},
}


def get_model_and_tokenizer(args):
    # # Tokenizer
    # if args.model_type.startswith("gpt2"):
    #     tokenizer = (
    #         AutoTokenizer.from_pretrained("gpt2")
    #         if args.tokenizer_type == "word"
    #         else CharTokenizer(args.seq_len)
    #     )

    #     # TODO: Check if necessary for LLAMA too
    #     if args.tokenizer_type == "word":
    #         tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    if args.tokenizer_type == "word":
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME_MAP[args.model_type]["model_name"]
        )
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
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
        model_kwargs={"hf_model_name": MODEL_NAME_MAP[args.model_type]["model_name"]},
    )

    # Add LLaMA specific config
    if args.model_type == "llama":
        model = TJDLLAMA(model_config)
    elif args.model_type == "gpt2":
        model = TJDGPT2(model_config)
    elif args.model_type == "gpt2r":
        model = GPT2(model_config)
    elif args.model_type == "llamar":
        model = LLAMA(model_config)
    else:
        raise ValueError(f"Model type {args.model_type} not recognized.")

    chat_template = {
        "sharegpt": ChatTemplateShareGPTV2,
        "shakespeare": ChatTemplateShakespeare,
        "gsm8k": ChatTemplateGSM8k,
        "stemp": ChatTemplateSynTemp,
        "snum": ChatTemplateSynNum,
        "sbase": ChatTemplateSynNumBase,
    }[args.dataset]

    return model, tokenizer, chat_template
