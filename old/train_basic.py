"""
Hardware Requirements (for Llama-based models):
    - GPUs: 4x NVIDIA A100 80GB GPUs
    - CPU RAM: 128GB minimum

    Note: GPT-2 based models require significantly less resources

Recommended SLURM allocation (for Llama):
    salloc --gres=gpu:a100l:4 --mem=128G --cpus-per-task=32

Usage:
    - accelerate launch --multi_gpu train.py ...

References:
    - HuggingFace multi-GPU training: https://huggingface.co/docs/transformers/en/perf_train_gpu_many
"""

# python train.py --model_type llama --model_head base --horizon 1 --horizon_eval 1 --dataset sharegpt --freeze_base_model --batch_size 2 --seq_len 32

import os
import os.path as osp
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import wandb

from peft import LoraConfig, TaskType, get_peft_model

from data.gsm8k import load_gsm8k_data
from data.shakespeare import load_shakespeare_data
from data.sharegptv2 import load_sharegptv2_data
from data.syn_number_bases import load_syn_num_base_data
from data.syn_numbers import load_syn_num_data
from data.syn_temp import load_syn_temp_data
from data.wikitext import load_wikitext_data
from utils.helpers import (
    get_test_samples,
    parse_args,
    set_seed,
)
from utils.utils import get_experiment_name


def main():

    args = parse_args()
    set_seed(args.seed)

    exp_name = get_experiment_name(vars(args))
    ckpt_dir = osp.join("checkpoints", exp_name)
    has_checkpoint = osp.exists(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # meta-llama/Llama-2-7b-chat-hf
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True
    )
    p_model = get_peft_model(
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
    # TODO remove this line
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
        "sharegpt": load_sharegptv2_data,
        "gsm8k": load_gsm8k_data,
        "stemp": load_syn_temp_data,
        "snum": load_syn_num_data,
        "sbase": load_syn_num_base_data,
    }["gsm8k"](tokenizer, args.seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        max_grad_norm=args.grad_clip_val,
        # Logging
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Evaluation
        # eval_on_start=True,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        # Reporting
        report_to="wandb",  # Disable wandb for eval only
        # Checkpoints
        save_strategy="best",  # Save model every epoch
        save_safetensors=False,
        save_total_limit=1,
        metric_for_best_model="eval_nll",
        greater_is_better=False,
        # remove_unused_columns=False,
        # Memory optimization
        # no_cuda=True,  # Force CPU usage
        # deepspeed=args.deepspeed_path,
        # # OPT 1.
        # bf16=True,  # Enable bfloat16 mixed precision
        # gradient_checkpointing=True,
        # gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        # optim="adamw_bnb_8bit",  # Use Adafactor optimizer
        # OPT 2.
        # fsdp="full_shard",
    )

    if training_args.local_rank == 0:  # main process
        project_name = "tjdnet-dev"
        wandb.init(
            project=f"train-basic-{project_name}",
            name=exp_name,
            id=args.wandb_id,
        )

    # Initialize the trainer
    trainer = Trainer(
        model=p_model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["eval"],
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint and has_checkpoint)

    # Save the model
    trainer.save_model(ckpt_dir)

    # Generate a test sample
    test_sample = get_test_samples(
        p_model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        prompt="What is the meaning of life?",
    )
    print(f"Test sample:\n{test_sample}")


if __name__ == "__main__":
    main()
