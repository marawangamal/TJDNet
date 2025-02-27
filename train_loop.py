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

import os.path as osp
import os

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model

from torch.utils.data import DataLoader
from accelerate import Accelerator

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
    scheduler=None,
):

    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        if scheduler:
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
    # model, tokenizer = get_model_and_tokenizer(args)
    # chat_template = get_chat_template(args)

    # meta-llama/Llama-2-7b-chat-hf
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=4,
            lora_alpha=32,
            lora_dropout=0.1,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Print dict key value pairs
    print("Model parameters:")
    print(
        "# Params (Trainable):",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("# Params (Total):", sum(p.numel() for p in model.parameters()))

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

    train_dataloader = DataLoader(
        lm_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        lm_dataset["eval"],
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Train
    train_loop(
        model=model,
        train_dataloader=train_dataloader,
        accelerator=accelerator,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    main()
