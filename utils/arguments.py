from dataloaders import DATASETS
from tjdnet.distributions import TJD_DISTS


import argparse


def add_train_args(parser: argparse.ArgumentParser):

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
    parser.add_argument(
        "--accum_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=[
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
            "64",
            "32",
            "16",
            "bf16",
        ],
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
    # TODO: rename to init_mode
    parser.add_argument(
        "--init_mode",
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
        choices=DATASETS.keys(),
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
        choices=[
            "base",
            "draft",
            "speculative",
            "draft_multi_horizon",
            "base_multi_horizon",
            "mixed",
        ],
        help="Generation mode for the model.",
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
        default=None,
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
        default="single",
        choices=["auto", "fsdp", "ddp", "deepspeed", "single"],
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to run the test set.",
        default=False,
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Whether to run a single batch for debugging.",
        default=False,
    )
    parser.add_argument(
        "--slurm_job_id",
        type=int,
        help="Slurm job ID for lookup.",
        default=None,
    )
    parser.add_argument(
        "--group_id",
        type=str,
        help="Group ID for jrun.",
        default=None,
    )
    parser.add_argument(
        "--group_level",
        type=int,
        default=0,
        help="Group level to filter the models",
    )
    parser.add_argument(
        "--lookup",
        action="store_true",
        help="Whether to lookup models in the group.",
        default=False,
    )
    parser.add_argument(
        "--delete_ckpt",
        action="store_true",
        help="Whether to delete the checkpoint after evaluation.",
        default=False,
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Index of the model to train in the group.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/mila/m/marawan.gamal/scratch/hf_cache",
        help="Path to the cache directory.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Interval for validation checks during training.",
    )


def add_test_args(parser: argparse.ArgumentParser):
    # ------------------
    # Basic arguments
    # ------------------
    parser.add_argument(
        "--experiment",
        type=str,
        help="Path to the checkpoint to evaluate.",
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Block size for model input sequences.",
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
        "--do_sample",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Type of dataset to use for evaluation.",
        choices=DATASETS.keys(),
    )
    parser.add_argument(
        "--template_mode",
        type=str,
        default=None,
        help="Template type to use for evaluation.",
        choices=["0_shot", "few_shot"],
    )

    # ---
    # MISC
    # ---
    parser.add_argument(
        "--group_id",
        type=str,
        help="Group ID for jrun.",
        default=None,
    )
    parser.add_argument(
        "--group_level",
        type=int,
        default=0,
        help="Group level to filter the models",
    )
    parser.add_argument(
        "--lookup",
        action="store_true",
        help="Whether to delete the checkpoint after evaluation.",
        default=False,
    )
    parser.add_argument(
        "--delete_ckpt",
        action="store_true",
        help="Whether to delete the checkpoint after evaluation.",
        default=False,
    )
    parser.add_argument(
        "--gen_mode",
        type=str,
        default="draft",
        choices=[
            "base",
            "draft",
            "speculative",
            "draft_multi_horizon",
            "base_multi_horizon",
            "mixed",
        ],
        help="Generation mode for the model.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Index of the model to train in the group.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/mila/m/marawan.gamal/scratch/hf_cache",
        help="Path to the cache directory.",
    )


# Loss modes:
# 1. joint - train the model with joint loss (target + draft)
# 2. draft - train the model with draft loss (draft only)

# Gen modes:
# 1. base - generate tokens one by one, no speculative decoding
# 2. draft - generate tokens in blocks, no speculative decoding
# 3. speculative - generate tokens in blocks, speculative decoding with draft model
# 4. mixed - generate and log base, draft and speculative decoding results


def add_tag_args(parser: argparse.ArgumentParser):
    # ------------------
    # Basic arguments
    # ------------------
    parser.add_argument(
        "--group_id",
        type=str,
        help="Group ID for jrun.",
        default=None,
    )
    parser.add_argument(
        "--group_level",
        type=int,
        default=0,
        help="Group level to filter the models",
    )
    parser.add_argument(
        "--group_by",
        nargs="+",
        help="Tag best within each group_by attr (can specify multiple attributes)",
    )


def parse_args():
    root = argparse.ArgumentParser(
        description="Train / evaluate TJD models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(dest="cmd", required=True)

    # ----- train -----
    train = sub.add_parser("train", parents=[], help="Fit the model")
    add_train_args(train)  # everything available

    # ----- test -----
    test = sub.add_parser("test", parents=[], help="Evaluate a checkpoint")
    add_test_args(test)

    # ----- tag -----
    tag = sub.add_parser("tag", parents=[], help="Tag a checkpoint")
    add_tag_args(tag)

    args = root.parse_args()
    # _validate_args(args)
    return args


def _validate_args(args):
    rules = [
        {
            "message": "Cannot use fsdp strategy during testing.",
            "condition": lambda: not hasattr(args, "accel_strategy")
            and (args.accel_strategy == "fsdp" and args.test),
        }
    ]

    for rule in rules:
        if not rule["condition"]():
            raise ValueError(rule["message"])


# python train_pl.py --accel_strategy ddp --dataset gsm8k --model meta-llama/Llama-3.2-3B-Instruct --epochs 10 --batch_size 1 --accum_grad_batches 8 --seq_len 128 --lr 5e-5 --model_head base --rank 1 --horizon 1
# python train_pl.py --accel_strategy fsdp --dataset gsm8k --model meta-llama/Llama-3.2-3B-Instruct --epochs 10 --batch_size 8 --seq_len 128 --lr 5e-5 --model_head base --rank 1 --horizon 1
