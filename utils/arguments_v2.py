from tjdnet.distributions import TJD_DISTS
from utils.helpers import validate_args


import argparse


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
        "--accel_strategy",
        type=str,
        default="auto",
        choices=["auto", "fsdp"],
    )

    args = parser.parse_args()
    validate_args(args)
    return parser.parse_args()
