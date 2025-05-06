"""Training script for TJDNet models.

This script trains and evaluates TJDNet models using the Hugging Face Transformers library.

Example:
    accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_2gpus.yaml train.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset gsm8k \
        --epochs 50 \
        --batch_size 8 \
        --seq_len 128 \
        --lr 1e-5 \
        --model_head cp \
        --hidden_dim 5120 \
        --horizon 2 \
        --horizon_eval 2 \
        --rank 16

Hardware requirements:
    - LLAMA 7B model: 4x GPUs w/ 80GB VRAM (FSDP)
    - GPT-2 model: 1x GPUs w/ 40GB VRAM

"""

from argparse import Namespace
import copy
import json
import os
import os.path as osp
from re import L
import shutil
import time
from typing import Literal, Union
import uuid

import torch
import wandb
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


from dataloaders import CHAT_TEMPLATES, DATASET_LOADERS
from dataloaders._base import BaseChatTemplate
from utils.accpetance_rates import compute_acceptance_rate
from utils.accuracy import compute_accuracy
from utils.utils import get_experiment_name
from utils.helpers import (
    get_git_info,
    get_model_and_tokenizer,
    parse_args,
    save_args,
    set_seed,
)

CHECKPOINT_DIR = "checkpoints"

EXP_NAME_EXCLUSIONS = ["cache_dir", "disable_wandb", "slurm_job_id"]


class TJDTrainer(Trainer):
    def __init__(
        self,
        test_dataset: torch.utils.data.Dataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        chat_template: BaseChatTemplate,
        generate_kwargs: dict,
        # horizon: int,
        # top_k: int,
        # eos_token: str,
        acc_batch_size: int = 1,
        on_converge_callback_cs=None,
        metric: Literal["accuracy", "acceptance_rate"] = "accuracy",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_dataset = test_dataset
        self.chat_template = chat_template

        # self.horizon = horizon
        # self.top_k = top_k
        # self.eos_token = eos_token
        self.tokenizer = tokenizer
        self.acc_batch_size = acc_batch_size
        self.generate_kwargs = generate_kwargs
        self.on_converge_callback_cs = on_converge_callback_cs
        self.metric_name = metric
        self.metric_fn = {
            "accuracy": compute_accuracy,
            "acceptance_rate": compute_acceptance_rate,
        }[metric]

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        output_dict = model(**inputs)
        loss = output_dict["loss"]
        return (loss, output_dict) if return_outputs else loss

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        if self.test_dataset:
            acc, _ = self.metric_fn(
                self.model,
                tokenizer=self.tokenizer,  # type: ignore
                test_dataset=self.test_dataset,  # type: ignore
                chat_template=self.chat_template,
                # horizon=self.horizon,
                # top_k=self.top_k,
                # eos_token=self.eos_token,
                batch_size=self.acc_batch_size,
                generate_kwargs=self.generate_kwargs,
            )

            if output and output.metrics:
                output.metrics[f"eval_acc"] = acc

            print(f"Eval {self.metric_name}:", acc)
            if abs(acc - 1.0) < 1e-3:
                print(f"Eval {self.metric_name} is 1.0, ending training.")
                if (
                    self.on_converge_callback_cs is not None
                    and output
                    and output.metrics
                ):
                    self.on_converge_callback_cs(
                        {**output.metrics, "epoch": self.state.epoch}
                    )
                exit(0)

        return output


def setup_dist_class_fsdp_wrapping(model, training_args):
    """Modify FSDP wrapping to include distribution classes."""
    import importlib
    from functools import partial

    # Find all distribution classes in the model
    dist_classes = set()

    def find_dist_classes_in_model(module):
        for name, child in module.named_children():
            if "Dist" in child.__class__.__name__:
                dist_classes.add(child.__class__)
                print(f"Found distribution class: {child.__class__.__name__}")
            find_dist_classes_in_model(child)

    # Find classes in the model
    find_dist_classes_in_model(model)

    # Only proceed if we found distribution classes and have FSDP
    if (
        dist_classes
        and hasattr(training_args, "_fsdp_plugin")
        and training_args._fsdp_plugin is not None
    ):
        # Get existing auto wrap policy
        fsdp_plugin = training_args._fsdp_plugin
        existing_policy = getattr(fsdp_plugin, "auto_wrap_policy", None)

        # Create a combined policy
        def combined_wrap_policy(module, recurse=True, **kwargs):
            # Check if module is one of our distribution classes
            if module.__class__ in dist_classes:
                return True

            # Otherwise, use the existing policy
            if callable(existing_policy):
                return existing_policy(module, recurse=recurse, **kwargs)

            return False

        # Apply the combined policy
        fsdp_plugin.auto_wrap_policy = combined_wrap_policy
        print(
            f"Applied custom FSDP wrapping for {len(dist_classes)} distribution classes"
        )
    else:
        print("No distribution classes found or FSDP not active")


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


def get_exp_config(exp_path):
    """Load the experiment configuration from a file."""
    if not osp.exists(osp.join(exp_path, "args.json")):
        return None
    with open(osp.join(exp_path, "args.json"), "r") as f:
        return json.load(f)


def lookup_wandb_id(args):
    exps = os.listdir(CHECKPOINT_DIR)
    args_cp = vars(args).copy()
    if "wandb_id" in args_cp:
        del args_cp["wandb_id"]
    matches = [exp for exp in exps if exp.startswith(get_experiment_name(args_cp))]
    exp_args = (
        get_exp_config(osp.join(CHECKPOINT_DIR, matches[0]))
        if len(matches) == 1
        else None
    )
    return exp_args["wandb_id"] if exp_args else None


def setup(args, local_rank: int):
    # mkdir for checkpoints if not exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    wandb_id = None
    iterations = 0
    while wandb_id is None:
        wandb_id = lookup_wandb_id(args)
        if local_rank == 0 and wandb_id is None:
            wandb_id = generate_wandb_id()
            args.wandb_id = wandb_id
            exp_name = get_experiment_name(vars(args))
            ckpt_dir = osp.join(CHECKPOINT_DIR, exp_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            save_args(args, ckpt_dir)
            print(f"[{local_rank}] Generated new wandb_id: {wandb_id}")
            print(f"[{local_rank}] lookup_wandb_id: {lookup_wandb_id(args)}")
        elif wandb_id is None:
            time.sleep(1)  # Sleep for a few seconds
        iterations += 1

        if iterations > 10:
            raise ValueError("Failed to find or generate a wandb_id")

        print(f"[{local_rank}] wandb_id: {wandb_id}")

    args.wandb_id = wandb_id
    exp_name = get_experiment_name(vars(args))
    ckpt_dir = osp.join(CHECKPOINT_DIR, exp_name)

    has_checkpoint = False
    if osp.exists(ckpt_dir):
        # Look for actual checkpoint files (like pytorch_model.bin or similar)
        checkpoint_files = [
            f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint-")
        ]

        # Fix files - delete if checkpoint is empty
        for f in checkpoint_files:
            if not "trainer_state.json" in os.listdir(osp.join(ckpt_dir, f)):
                try:
                    print(f"Deleting corrupt checkpoint: {f}")
                    shutil.rmtree(osp.join(ckpt_dir, f))
                    checkpoint_files.remove(f)
                except Exception as e:
                    print(f"Error deleting checkpoint {f}: {e}")

        # Check if there are any checkpoint files
        has_checkpoint = len(checkpoint_files) > 0
        if has_checkpoint:
            print(f"Resuming from checkpoint: {ckpt_dir}")
        else:
            print(f"No checkpoint found in {ckpt_dir}. Starting fresh.")

    return args, exp_name, ckpt_dir, has_checkpoint


def main():
    # Configuration
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args_raw = parse_args()
    filtered_args = Namespace(
        **{k: v for k, v in vars(args_raw).items() if k not in EXP_NAME_EXCLUSIONS}
    )
    args, exp_name, ckpt_dir, has_checkpoint = setup(filtered_args, local_rank)
    set_seed(42)

    # Model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)
    chat_template = CHAT_TEMPLATES[args.dataset]

    params_dict = model.param_dict
    # Print dict key value pairs
    print("Model parameters:")
    print("\n".join([f"{k}: {v}" for k, v in params_dict.items()]))

    # Datasets
    lm_dataset = DATASET_LOADERS[args.dataset](
        tokenizer=tokenizer,
        input_seq_len=args.seq_len,
        max_num_samples=args.max_num_samples,
        print_stats=local_rank == 0,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def on_converge_callback_cs(metrics):
        # save metrics to to accuracy.json
        accuracy_file = os.path.join(ckpt_dir, "eval_converged_metrics.json")
        with open(accuracy_file, "w") as f:
            json.dump(metrics, f)
        print(f"Accuracy saved to {accuracy_file}")

    # Check if exp previously ran and converged
    if has_checkpoint:
        # Check if accuracy.json exists
        accuracy_file = os.path.join(ckpt_dir, "eval_converged_metrics.json")
        if os.path.exists(accuracy_file):
            with open(accuracy_file, "r") as f:
                metrics = json.load(f)
            print(f"Converged metrics: {metrics}")
            if (
                "eval_acc" in metrics
                and abs(metrics["eval_acc"] - 1.0) < 1e-3
                and local_rank == 0
            ):
                print("Accuracy is 1.0, ending training.")
                exit(0)

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
        report_to="wandb" if not args_raw.disable_wandb else "none",
        # Checkpoints
        # prev save_strategy ===>
        # save_strategy=args.eval_strategy,
        # save_steps=args.eval_steps,
        # save_total_limit=2,  # Save only 3 checkpoints
        # load_best_model_at_end=True,
        # ====
        save_strategy="best",
        # <=== new save_strategy
        save_safetensors=False,
        metric_for_best_model="eval_nll",
        greater_is_better=False,
        # remove_unused_columns=False,
        # Memory optimization
        # fp16=True,  # Enable bfloat16 mixed precision
        # gradient_checkpointing=True,
        # gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        optim="adafactor",  # Use Adafactor optimizer
        # torch_empty_cache_steps=1,
        # no_cuda=True,  # Force CPU usage
    )

    if training_args.local_rank == 0:  # main process
        git_info = get_git_info()
        project_name = (
            "tjdnet-prod" if git_info.get("branch") == "main" else "tjdnet-dev"
        )
        if not args_raw.disable_wandb:
            wandb.init(
                project=project_name,
                name=exp_name,
                id=args.wandb_id,
                config={**vars(args), **git_info},
                resume="allow",
            )

    # In your main function, add this before initializing the trainer:
    # generation_callback = GenerationCallback(
    #     model=model,
    #     tokenizer=tokenizer,
    #     generate_strategy=args.generate_strategy,
    #     generate_steps=args.generate_steps,  # or any other frequency you want
    #     max_new_tokens=args.max_new_tokens,
    #     horizon=args.horizon_eval,
    #     chat_template=chat_template,
    #     top_k=args.top_k,
    #     disable_wandb=args_raw.disable_wandb,
    # )

    # Add this line here:
    setup_dist_class_fsdp_wrapping(model, training_args)

    # Initialize the trainer
    trainer = TJDTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[generation_callback] if args.compute_acc else None,
        # Evaluation
        tokenizer=tokenizer,
        test_dataset=lm_dataset["test"] if args.compute_acc else None,  # type: ignore
        chat_template=chat_template,
        generate_kwargs=dict(
            horizon=args.horizon_eval,
            top_k=args.top_k,
            top_p=0.95,
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            stop_token=tokenizer.eos_token_id,
        ),
        acc_batch_size=args.acc_batch_size,
        on_converge_callback_cs=on_converge_callback_cs,
        metric="acceptance_rate" if args.use_speculative_sampling else "accuracy",
    )

    if args.eval_only:
        # Run evaluation only
        metrics = trainer.evaluate()
        print("Evaluation metrics:", metrics)
    else:
        trainer.train(resume_from_checkpoint=has_checkpoint)

        # Save the model
        trainer.save_model(ckpt_dir)


if __name__ == "__main__":
    main()
