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

import os
import os.path as osp
import uuid

import wandb
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


from utils.accuracy import compute_accuracy
from utils.generation import GenerationCallback
from data.gsm8k import load_gsm8k_data
from data.shakespeare import load_shakespeare_data
from data.sharegptv2 import load_sharegptv2_data
from data.syn_number_bases import load_syn_num_base_data
from data.syn_numbers import load_syn_num_data
from data.syn_temp import load_syn_temp_data
from data.wikitext import load_wikitext_data
from utils.utils import get_experiment_name
from utils.helpers import (
    get_chat_template,
    get_git_info,
    get_model_and_tokenizer,
    get_test_samples,
    parse_args,
    save_args,
    set_seed,
)


class TJDTrainer(Trainer):
    def __init__(
        self,
        test_dataset,
        tokenizer,
        chat_template,
        horizon,
        top_k,
        num_beams,
        eos_token,
        acc_batch_size=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_dataset = test_dataset
        self.chat_template = chat_template

        self.horizon = horizon
        self.top_k = top_k
        self.num_beams = num_beams
        self.eos_token = eos_token
        self.tokenizer = tokenizer
        self.acc_batch_size = acc_batch_size

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        output_dict = model(**inputs)
        loss = output_dict["loss"]
        return (loss, output_dict) if return_outputs else loss

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        # TODO: Use dataloader instead of dataset
        # TODO: Refactor -- utils/callbacks/eval_gsm8k.py ==> utils/accuracy.py
        if self.test_dataset:
            acc = compute_accuracy(
                self.model,
                tokenizer=self.tokenizer,
                test_dataset=self.test_dataset,
                chat_template=self.chat_template,
                horizon=self.horizon,
                top_k=self.top_k,
                batch_size=self.acc_batch_size,
                # eos_token=self.eos_token,
            )
            print("Eval accuracy:", acc)
            if output and output.metrics:
                output.metrics[f"eval_acc"] = acc
        return output


# Custom evaluation function
def compute_metrics(eval_pred):
    # Note: If return type of model forward is a dict, then the `predictions` will be tuple of all vals of keys except loss
    # See `prediction_step` in Trainer class
    (nll, loss_scale), labels = eval_pred
    return {
        "nll": nll.mean().item(),
    }


def generate_wandb_id():
    """Generate a random wandb_id that's compatible with W&B requirements."""
    # Generate a random UUID and take the first 8 characters
    # This gives us plenty of uniqueness while keeping the ID short
    random_id = str(uuid.uuid4()).replace("-", "")[:8]
    return random_id


def main():
    # Configuration
    args = parse_args()
    if hasattr(args, "wandb_id") and args.wandb_id is None:
        args.wandb_id = generate_wandb_id()
    exp_name = get_experiment_name(vars(args))
    ckpt_dir = osp.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    has_checkpoint = False
    if osp.exists(ckpt_dir):
        # Look for actual checkpoint files (like pytorch_model.bin or similar)
        checkpoint_files = [
            f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint-")
        ]
        has_checkpoint = len(checkpoint_files) > 0
        if has_checkpoint:
            print(f"Resuming from checkpoint: {ckpt_dir}")

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
    # data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

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
        # report_to="none" if args.eval_only else "wandb",  # Disable wandb for eval only
        report_to="wandb",
        # Checkpoints
        save_strategy="best",  # Save model every epoch
        save_total_limit=1,
        save_safetensors=False,
        metric_for_best_model="eval_nll",
        greater_is_better=False,
        # remove_unused_columns=False,
        # Memory optimization
        # fp16=True,  # Enable bfloat16 mixed precision
        # gradient_checkpointing=True,
        # gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        # optim="adafactor",  # Use Adafactor optimizer
        # torch_empty_cache_steps=1,
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
            id=args.wandb_id,
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

    # Initialize the trainer
    trainer = TJDTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[generation_callback] if args.compute_acc else None,
        # Evaluation
        tokenizer=tokenizer,
        test_dataset=lm_dataset["test"] if args.compute_acc else None,
        chat_template=chat_template,
        horizon=args.horizon_eval,
        top_k=args.top_k,
        num_beams=args.num_beams,
        eos_token=(
            tokenizer.eos_token
            if args.tokenizer_type == "word"
            else tokenizer.sep_token
        ),
        acc_batch_size=args.acc_batch_size,
    )

    if args.eval_only:
        # Run evaluation only
        metrics = trainer.evaluate()
        print("Evaluation metrics:", metrics)
    else:
        trainer.train(resume_from_checkpoint=has_checkpoint)

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
