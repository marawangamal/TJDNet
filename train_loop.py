"""
Hardware Requirements (for Llama-based models):
    - GPUs: 4x NVIDIA A100 80GB GPUs 
    - CPU RAM: 128GB minimum
    - Storage: Recommend 1TB+ SSD for dataset and checkpoints

    Note: GPT-2 based models require significantly less resources

Recommended SLURM allocation (for Llama):
    salloc --gres=gpu:a100l:4 --mem=128G --cpus-per-task=32

Usage:
    - Uses PyTorch Distributed Data Parallel (DDP) for multi-GPU training
    - Automatic mixed precision (AMP) enabled for memory efficiency
    - Gradient checkpointing available for large models
    
References:
    - HuggingFace multi-GPU training: https://huggingface.co/docs/transformers/en/perf_train_gpu_many
"""

# python train.py --model_type llama --model_head base --horizon 1 --horizon_eval 1 --dataset sharegpt --freeze_base_model --batch_size 2 --seq_len 32

import os.path as osp
import os

import torch
from transformers import (
    DataCollatorForLanguageModeling,
)

from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType

from data.gsm8k import load_gsm8k_data
from data.shakespeare import load_shakespeare_data
from data.sharegptv2 import load_sharegptv2_data
from data.syn_number_bases import load_syn_num_base_data
from data.syn_numbers import load_syn_num_data
from data.syn_temp import load_syn_temp_data
from data.wikitext import load_wikitext_data
from utils import get_experiment_name
from helpers import (
    get_chat_template,
    get_model_and_tokenizer,
    parse_args,
    save_args,
    set_seed,
)


def train_loop(
    model,
    train_dataloader,
    accelerator,
    optimizer,
    scheduler,
):

    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()


def main():
    # Configuration
    args = parse_args()
    exp_name = get_experiment_name(vars(args))
    # Add timestamp to exp_name
    # exp_name += f"_{int(time.time())}"  -- remove to facilitate resume_from_checkpoint
    ckpt_dir = osp.join("checkpoints", exp_name)
    has_checkpoint = osp.exists(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)
    save_args(args, ckpt_dir)

    # Model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)
    chat_template = get_chat_template(args)

    params_dict = model.param_dict
    # Print dict key value pairs
    print("Model parameters:")
    print("\n".join([f"{k}: {v}" for k, v in params_dict.items()]))

    # Datasets
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
        "sharegpt": load_sharegptv2_data,
        "gsm8k": load_gsm8k_data,
        "stemp": load_syn_temp_data,
        "snum": load_syn_num_data,
        "sbase": load_syn_num_base_data,
    }[args.dataset](tokenizer, args.seq_len, max_num_samples=args.max_num_samples)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        lm_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        lm_dataset["eval"],
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes
        ),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    train_loop(model, tokenizer, train_dataloader, eval_dataloader, optimizer)


if __name__ == "__main__":
    main()
