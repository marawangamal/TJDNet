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
import json
import os
import os.path as osp
from re import L
import shutil
import time
from typing import Literal, Union

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
from tjdnet.models.tjd import TJDGenerationConfig
from utils.accpetance_rates import compute_acceptance_rate
from utils.accuracy import compute_accuracy
from utils.utils import get_experiment_name
from utils.helpers import (
    get_git_info,
    get_model_and_tokenizer_v2,
    parse_args,
    save_args,
    set_seed,
)
from utils.utils import printo
from utils.utils import printr
from utils.utils import generate_wandb_id

CHECKPOINT_DIR = "checkpoints"
SILENT_ARGS = [
    "slurm_job_id",
    "cache_dir",
    "disable_wandb",
    "compute_acc",
    "generate_strategy",
    "generate_steps",
    "logging_strategy",
    "logging_steps",
    "eval_strategy",
    "eval_steps",
    "wandb_project",
    # Evaluation args
    "do_sample",
    "max_new_tokens",
    "top_k",
    "gen_mode",
]


class TJDTrainer(Trainer):
    def __init__(
        self,
        test_dataset: torch.utils.data.Dataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        chat_template: BaseChatTemplate,
        generation_config: TJDGenerationConfig,
        on_converge_callback_cs=None,
        metric: Literal["accuracy", "acceptance_rate"] = "accuracy",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_dataset = test_dataset
        self.chat_template = chat_template
        self.tokenizer = tokenizer
        self.generation_config = generation_config
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
                generation_config=self.generation_config,
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


# Custom evaluation function
def compute_metrics(eval_pred):
    # Note: If return type of model forward is a dict, then the `predictions` will be tuple of all vals of keys except loss
    # See `prediction_step` in Trainer class
    (nll, loss_scale), labels = eval_pred
    return {
        "nll": nll.mean().item(),
    }


def get_wandb_id(exp_name):
    exps = os.listdir(CHECKPOINT_DIR)
    matches = [exp for exp in exps if exp.startswith(exp_name)]
    if len(matches) == 1:
        args_file = os.path.join(CHECKPOINT_DIR, matches[0], "args.json")
        if osp.exists(args_file):
            with open(osp.join(args_file), "r") as f:
                printr(f"Found args file: {args_file}")
                return json.load(f)["wandb_id"]
        else:
            wandb_id = generate_wandb_id()
            printr(f"Generated new wandb_id: {wandb_id}")
            return wandb_id
    else:
        return None


def has_valid_checkpoint(ckpt_dir):
    # Remove faulty checkpoints
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
            printr(f"Resuming from checkpoint: {ckpt_dir}")
        else:
            printr(f"No checkpoint found in {ckpt_dir}. Starting fresh.")

    return has_checkpoint


def main():
    # ==== Setup
    set_seed(42)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args = parse_args()
    filtered_args = Namespace(
        **{k: v for k, v in vars(args).items() if k not in SILENT_ARGS}
    )
    # exp name does not include silent args
    exp_name = get_experiment_name(vars(filtered_args))
    ckpt_dir = osp.join(CHECKPOINT_DIR, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Sync file path
    sync_file = os.path.join(ckpt_dir, ".rank0_done")

    # Rank 0 does initialization
    if local_rank == 0:
        # Clean up old flag if it exists
        if os.path.exists(sync_file):
            os.remove(sync_file)

        wandb_id = get_wandb_id(exp_name)
        args.wandb_id = wandb_id
        save_args(args, ckpt_dir)
        has_checkpoint = has_valid_checkpoint(ckpt_dir)

        # Signal completion by creating an empty file
        with open(sync_file, "w") as f:
            pass  # Create empty file

    # Other ranks wait for the file to appear
    else:
        wait_time = 0
        while not os.path.exists(sync_file):
            if wait_time > 10:
                raise ValueError("Setup failed")
            time.sleep(1)
            printr("Waiting for rank 0...")

        # Load the args file
        with open(os.path.join(ckpt_dir, "args.json"), "r") as f:
            saved_args = json.load(f)
            args.wandb_id = saved_args.get("wandb_id")

        has_checkpoint = has_valid_checkpoint(ckpt_dir)

        # Optional: Clean up the sync file when all processes are past the setup
        # Use a distributed barrier if available (preferred method)
        if os.path.exists(sync_file) and local_rank == 1:
            printo("Cleaning up...")
            os.remove(sync_file)
        printr("Setup complete.")
    # ====

    # 1. Model and tokenizer
    printr("Initializing model...")
    model, tokenizer = get_model_and_tokenizer_v2(args)
    chat_template = CHAT_TEMPLATES[args.dataset]
    printo(f"Model: {model.__class__.__name__}")
    printo(f"Tokenizer: {tokenizer.__class__.__name__}")

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
                print(f"Experiment: {ckpt_dir})")
                return

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
        report_to="wandb" if not args.disable_wandb else "none",
        # Checkpoints
        # prev save_strategy ===>
        # save_strategy=args.eval_strategy,
        # save_steps=args.eval_steps,
        # save_total_limit=2,  # Save only 3 checkpoints
        # load_best_model_at_end=True,
        # == new save_strategy ==>
        save_strategy="best",
        save_total_limit=1,
        # <====
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
        suffix = "main" if git_info.get("branch") == "main" else "dev"
        project_name = f"{args.wandb_project}-{suffix}"
        if not args.disable_wandb:
            wandb.init(
                project=project_name,
                name=exp_name,
                id=args.wandb_id,
                config={**vars(args), **git_info},
                resume="allow",
            )

    # Initialize the trainer
    trainer = TJDTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # Evaluation
        tokenizer=tokenizer,
        test_dataset=lm_dataset["test"] if args.compute_acc else None,  # type: ignore
        chat_template=chat_template,
        generation_config=TJDGenerationConfig(
            do_sample=args.do_sample,
            horizon=args.horizon,
            max_new_tokens=args.max_new_tokens,
            gen_mode=args.gen_mode,
            top_k=args.top_k,
        ),
        on_converge_callback_cs=on_converge_callback_cs,
        # metric="acceptance_rate" if args.use_speculative_sampling else "accuracy",
    )
    printr(f"Trainer initialized with {len(lm_dataset['train'])} training samples")
    trainer.train(resume_from_checkpoint=has_checkpoint)
    print(f"Experiment: {ckpt_dir})")


if __name__ == "__main__":
    main()
