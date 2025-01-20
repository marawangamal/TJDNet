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
import wandb

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)


from data.shakespeare import load_shakespeare_data
from data.sharegpt import load_sharegpt_data
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


class GenerationCallback(TrainerCallback):
    def __init__(
        self,
        model,
        tokenizer,
        generate_strategy="steps",
        generate_steps=1000,
        max_new_tokens=100,
        horizon=1,
        prompt_formatter_fn=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generate_strategy = generate_strategy
        self.generate_steps = generate_steps
        self.max_new_tokens = max_new_tokens
        self.horizon = horizon
        self.prompts = [
            'complete the following code from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """',
        ]
        if prompt_formatter_fn is not None:
            self.prompts = [prompt_formatter_fn(prompt) for prompt in self.prompts]

    def on_step_end(self, args, state, control, **kwargs):
        if not args.local_rank == 0:
            return

        should_generate = False
        if self.generate_strategy == "steps":
            should_generate = state.global_step % self.generate_steps == 0
        elif self.generate_strategy == "epoch":
            # Check if we're at the end of an epoch
            should_generate = state.global_step % state.num_train_epochs == 0
        elif self.generate_strategy == "no":
            should_generate = False

        if should_generate:
            print("\n=== Generation Sample at step", state.global_step, "===")
            self.model.eval()

            samples = {}
            for i, prompt in enumerate(self.prompts):
                sample = get_test_samples(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    horizon=self.horizon,
                )
                samples[f"prompt_{i+1}"] = prompt
                samples[f"generation_{i+1}"] = sample
                print(f"\nPrompt: {prompt}\nOutput: {sample}\n")
                wandb.log(
                    {f"generation_text_{i}": wandb.Html(f"<pre>{sample}</pre>")},
                    step=state.global_step,
                )

            self.model.train()
            print("=" * 50 + "\n")


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
    }[args.dataset](tokenizer, args.seq_len, max_num_samples=args.max_num_samples)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        # Memory optimization
        # bf16=True,  # Enable bfloat16 mixed precision
        # gradient_checkpointing=True,
        # gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        # optim="adafactor",  # Use Adafactor optimizer
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
        prompt_formatter_fn=chat_template.format_prompt,
    )

    # Initialize the trainer
    trainer = TJDTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[generation_callback],  # Add this line
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
