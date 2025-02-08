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
import time
import torch
import wandb

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


from callbacks.eval_gsm8k import EvalGSM8KCallback
from callbacks.generation import GenerationCallback
from data.gsm8k import load_gsm8k_data
from data.shakespeare import load_shakespeare_data
from data.sharegpt import load_sharegpt_data
from data.syn_templates import load_synthetic_data
from data.wikitext import load_wikitext_data
from utils import get_experiment_name
from helpers import (
    get_git_info,
    get_model_and_tokenizer,
    get_test_samples,
    parse_args,
    save_args,
    set_seed,
)


class TJDTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        output_dict = model(**inputs)
        loss = output_dict["loss"]
        return (loss, output_dict) if return_outputs else loss


# Custom evaluation function
def compute_metrics(eval_pred):
    # Note: If return type of model forward is a dict, then the `predictions` will be tuple of all vals of keys except loss
    # See `prediction_step` in Trainer class
    (nll, loss_scale), labels = eval_pred
    return {
        "nll": nll.mean().item(),
    }


def main():
    # Configuration
    args = parse_args()
    exp_name = get_experiment_name(vars(args))
    # Add timestamp to exp_name
    exp_name += f"_{int(time.time())}"
    ckpt_dir = osp.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)
    save_args(args, ckpt_dir)

    # Model and tokenizer
    model, tokenizer, chat_template = get_model_and_tokenizer(args)
    params_dict = model.param_dict
    # Print dict key value pairs
    print("Model parameters:")
    print("\n".join([f"{k}: {v}" for k, v in params_dict.items()]))

    # Datasets
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
        "sharegpt": load_sharegpt_data,
        "gsm8k": load_gsm8k_data,
        "syn": load_synthetic_data,
    }[args.dataset](tokenizer, args.seq_len, max_num_samples=args.max_num_samples)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

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
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Evaluation
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        # Reporting
        report_to="wandb",  # Enable wandb logging
        # Checkpoints
        save_strategy="best",  # Save model every epoch
        save_safetensors=False,
        save_total_limit=1,
        metric_for_best_model="eval_nll",
        greater_is_better=False,
        # remove_unused_columns=False,
        # Memory optimization
        # bf16=True,  # Enable bfloat16 mixed precision
        # gradient_checkpointing=True,
        # gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        # optim="adafactor",  # Use Adafactor optimizer
        # no_cuda=True,  # Force CPU usage
    )

    if training_args.local_rank == 0:  # main process
        git_info = get_git_info()
        project_name = (
            "tjdnet-prod" if git_info.get("branch") == "main" else "tjdnet-dev"
        )
        wandb.init(
            project=project_name,
            name=exp_name,
        )

    # In your main function, add this before initializing the trainer:
    generation_callback = GenerationCallback(
        model=model,
        tokenizer=tokenizer,
        generate_strategy=args.generate_strategy,
        generate_steps=args.generate_steps,  # or any other frequency you want
        max_new_tokens=args.max_new_tokens,
        horizon=args.horizon_eval,
        chat_template=chat_template,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )
    eval_callback = (
        EvalGSM8KCallback(
            # TODO: fix this should always just be EOS token?
            test_dataset=lm_dataset["test_ungrouped"],
            eos_token=(
                tokenizer.eos_token
                if args.tokenizer_type == "word"
                else tokenizer.sep_token
            ),
            chat_template=chat_template,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            horizon=args.horizon_eval,
            num_beams=args.num_beams,
            tokenizer=tokenizer,
        )
        if args.dataset in ["gsm8k", "syn"]
        else None
    )

    # Initialize the trainer
    trainer = TJDTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[c for c in [generation_callback, eval_callback] if c is not None],
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(ckpt_dir)

    # Generate a test sample
    test_sample = get_test_samples(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        prompt="What is the meaning of life?",
    )
    print(f"Test sample:\n{test_sample}")


if __name__ == "__main__":
    main()
