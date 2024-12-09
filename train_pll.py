"""
Fine-tune GPT-2 using TJDNet on Shakespeare dataset.

Example usage:
python train_clm_shakespeare_char_tjd.py --model cpgpt2 

Resources: 
https://huggingface.co/docs/transformers/tasks/language_modeling
https://github.com/dariush-bahrami/character-tokenizer/tree/master


Two options for data loading:

Given a dataset of sequences of different length {s1, s2, ..., s2}, we have two options for dataloading

1. Simple (preprocess_simple)
    - Convert each sequence to be of length `max_len` via padding or trunction 

2. Advanced (preprocess_function & group texts)
    - Combine to sinlge length string s = [s_1, s_2, ..., s_b], then split into chunks of size `max_len`. This is less 
    - Less wastefulness from truncation


"""

from math import e
from typing import Dict, Any
import os.path as osp
import os
import numpy as np
import random
import argparse
import wandb

import torch
from transformers import DataCollatorForLanguageModeling, get_scheduler
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

from data.shakespeare import load_shakespeare_data
from data.wikitext import load_wikitext_data
from models.tjdgpt2.tjdgpt2 import TJDGPT2
from models.tjdgpt2.char_tokenizer import CharTokenizer
from utils import get_experiment_name


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on the ELI5 dataset.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
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
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--grad_clip_val",
        type=float,
        default=None,
        help="Gradient clipping value for training.",
    )
    parser.add_argument(
        "--scale_loss",
        default=False,
        action="store_true",
        help="Whether to scale the loss during training.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=384,
        help="Dimensionality of the model embeddings.",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=6,
        help="Number of hidden layers in the transformer model.",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=6,
        help="Number of attention heads in the transformer model.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--model",
        type=str,
        default="cp",
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
        "--dataset",
        type=str,
        default="shakespeare",
        help="Type of dataset to use for training.",
        choices=[
            "shakespeare",
            "wikitext",
        ],
    )
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
        default=4,
        help="Rank of the tensor train decomposition.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Block size for model input sequences.",
    )
    # Evaluation only arguments
    parser.add_argument(
        "--horizon_eval",
        type=int,
        default=1,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate during evaluation.",
    )
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


def get_test_samples(
    model,
    tokenizer,
    prompt="\n",
    max_new_tokens=8,
    # top_k=200,
    # temperature=0.8,
    num_beams=1,
    do_sample=False,
    horizon_eval=1,
    n_samples=1,
    print_output=True,
):
    # Inference
    model.eval()
    samples = []
    for i in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            inputs,
            num_beams=num_beams,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            horizon=horizon_eval,
        )
        sample = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if n_samples == 1:
            samples.append(sample)
        else:
            samples.append(f"[{i+1}] {sample}")

    if print_output:
        print("\n---\n".join(samples) + "\n")
    return "\n".join(samples)


class TJDTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        output_dict = model(**inputs)
        loss = output_dict["loss"]
        return (loss, output_dict) if return_outputs else loss


def main():
    args = parse_args()
    set_seed(args.seed)

    # Configuration
    exp_name = get_experiment_name(vars(args))
    ckpt_dir = osp.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Tokenizer
    tokenizer = (
        AutoTokenizer.from_pretrained("gpt2")
        if args.tokenizer_type == "word"
        else CharTokenizer(args.seq_len)
    )
    if args.tokenizer_type == "word":
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
    }[args.dataset](tokenizer, args.seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Model configuration
    model_config = {
        "model": args.model,
        "vocab_size": (
            len(tokenizer.get_vocab())
            if hasattr(tokenizer, "get_vocab")
            else len(tokenizer)
        ),
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "dropout": args.dropout,
        "rank": args.rank,
        "horizon": args.horizon,
        "positivity_func": args.positivity_func,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    model = TJDGPT2(**model_config)
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        max_grad_norm=args.grad_clip_val,
        eval_on_start=True,
        # Logging
        logging_strategy="steps",  # When to log: "steps", "epoch", or "no"
        logging_steps=100,  # Log every N steps (if strategy="steps")
        logging_first_step=True,  # Log the first step
        # Checkpoints
        save_strategy="no",  # Disable saving
        # Evaluation
        eval_strategy="epoch",  # Evaluate every epoch
        # Reporting
        report_to="wandb",  # Enable wandb logging
    )

    # Initialize wandb only on main process
    if training_args.local_rank == 0:  # main process
        wandb.init(
            project="tjdnet-shakepeare-prod",
            name=exp_name,
        )

    # Custom evaluation function
    def compute_metrics(eval_pred):
        # Note: If return type of model forward is a dict, then the `predictions` will be tuple of all vals of keys except loss
        # See `prediction_step` in Trainer class
        (nll, loss_scale), labels = eval_pred
        return {
            "nll": nll.mean().item(),
        }

    # Initialize the trainer
    trainer = TJDTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Generate a test sample
    test_sample = get_test_samples(model, tokenizer, max_new_tokens=args.max_new_tokens)
    print(f"Test sample:\n{test_sample}")


if __name__ == "__main__":
    main()
