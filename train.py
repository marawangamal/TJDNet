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
import functools
import gc
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from dataloaders import CHAT_TEMPLATES, DATASET_LOADERS
from dataloaders._base import BaseChatTemplate
from tjdnet.models.tjd import TJDGenerationConfig
from utils.accuracy import compute_accuracy
from utils.arguments import parse_args
from utils.monitor import log_memory
from utils.monitor import calculate_model_memory_breakdown
from utils.experiment_naming import get_experiment_name
from utils.helpers import (
    get_git_info,
    get_model_and_tokenizer_nowrap,
    get_model_and_tokenizer_tjdllama,
    get_model_and_tokenizer,
    get_model_and_tokenizer_tjdhfv2,
    save_args,
    set_seed,
)
from utils.utils import printo
from utils.utils import printr
from utils.utils import generate_wandb_id

EXPERIMENTS_DIR = "experiments"
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


def log_params(msg, model, rank=None):
    """Log model parameters at different stages."""
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", 0))

    # Only log from rank 0 to avoid cluttering logs
    if rank == 0:
        # Force garbage collection first
        gc.collect()
        torch.cuda.empty_cache()

        # Get model parameters
        total_params = sum(p.numel() for p in model.parameters())
        cuda_params = sum(
            p.numel() for p in model.parameters() if p.device.type == "cuda"
        )
        print(f"\n[PARAMS] - {msg}")
        print(f"  CUDA parameters: {cuda_params/1e9:.2f} B")
        print(f"  Total parameters: {total_params/1e9:.2f} B")
        print(
            f"  Parameter Sharing: {cuda_params/total_params:.2%} of params on this GPU"
        )


def get_layer_sizes(model):
    """Analyze sizes of different layers in the model."""
    layer_sizes = {}

    for name, param in model.named_parameters():
        # Get the top-level module name
        top_module = name.split(".")[0]
        if top_module not in layer_sizes:
            layer_sizes[top_module] = 0
        layer_sizes[top_module] += param.numel()
    return layer_sizes


class TJDTrainer(Trainer):
    def __init__(
        self,
        test_dataset: torch.utils.data.Dataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        chat_template: BaseChatTemplate,
        generation_config: TJDGenerationConfig,
        on_converge_callback_cs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_dataset = test_dataset
        self.chat_template = chat_template
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.on_converge_callback_cs = on_converge_callback_cs
        # self.metric_name = metric
        # self.metric_fn = {
        #     "accuracy": compute_accuracy,
        #     "acceptance_rate": compute_acceptance_rate,
        # }[metric]

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        output_dict = model(**inputs)
        loss = output_dict["loss"]
        return (loss, output_dict) if return_outputs else loss

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        if self.test_dataset:
            metrics = compute_accuracy(
                model=self.model,  # type: ignore
                tokenizer=self.tokenizer,  # type: ignore
                dataset=self.test_dataset,  # type: ignore
                chat_template=self.chat_template,
                generation_config=self.generation_config,
                max_iters=10,
            )

            if output and output.metrics:
                for k, v in metrics.items():
                    output.metrics[f"eval_{k}"] = v

            # acc = metrics.get("accuracy", 0.0)
            # print(f"Eval {self.metric_name}:", acc)
            # if abs(acc - 1.0) < 1e-3:
            #     print(f"Eval {self.metric_name} is 1.0, ending training.")
            #     if (
            #         self.on_converge_callback_cs is not None
            #         and output
            #         and output.metrics
            #     ):
            #         self.on_converge_callback_cs(
            #             {**output.metrics, "epoch": self.state.epoch}
            #         )
            #     exit(0)

        return output

    # def training_step(self, *args, **kwargs):
    #     # Log memory before training_step on first step
    #     if self.state.global_step == 0:
    #         log_memory(f"before_first_step", self.args.local_rank)

    #     result = super().training_step(*args, **kwargs)

    #     if self.state.global_step == 0:
    #         log_memory(f"after_first_step", self.args.local_rank)
    #     elif self.state.global_step == 1:
    #         log_memory(f"after_second_step", self.args.local_rank)
    #     elif self.state.global_step % 100 == 0:
    #         log_memory(f"step_{self.state.global_step}", self.args.local_rank)

    #     return result


class MemoryLoggingTJDTrainer(TJDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flag to track if first step has been logged
        self._logged_first_step = False

    def training_step(self, *args, **kwargs):
        """Log memory during backward pass."""
        # Call original training_step
        loss = super().training_step(*args, **kwargs)
        if self.state.global_step in [0, 1, 10, 20]:
            log_memory(
                f"batch {self.state.global_step} (before forward)", self.args.local_rank
            )
        return loss


# Custom evaluation function
def compute_metrics(eval_pred):
    # Note: If return type of model forward is a dict, then the `predictions` will be tuple of all vals of keys except loss
    # See `prediction_step` in Trainer class
    (nll, loss_draft, loss_target), labels = eval_pred
    return {
        "nll": nll.mean().item(),
        "loss_draft": loss_draft.mean().item(),
        "loss_target": loss_target.mean().item(),
    }


def get_wandb_id(exp_name):
    exps = os.listdir(EXPERIMENTS_DIR)
    matches = [exp for exp in exps if exp.startswith(exp_name)]
    if len(matches) == 1:
        args_file = os.path.join(EXPERIMENTS_DIR, matches[0], "args.json")
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
    # Rank 0 looks up the wandb id and creates a new if it doesn't exist
    set_seed(42)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args = parse_args()
    filtered_args = Namespace(
        **{k: v for k, v in vars(args).items() if k not in SILENT_ARGS}
    )
    # exp name does not include silent args
    exp_name = get_experiment_name(vars(filtered_args))
    ckpt_dir = osp.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Sync file path
    has_checkpoint = has_valid_checkpoint(ckpt_dir)

    # Rank 0 does initialization
    if local_rank == 0:
        wandb_id = get_wandb_id(exp_name)
        args.wandb_id = wandb_id
        save_args(args, ckpt_dir)
        printo(f"Setup complete.")

    # ====

    # 1. Model and tokenizer
    printr("Initializing model...")
    model, tokenizer = get_model_and_tokenizer(args)
    # model, tokenizer = get_model_and_tokenizer_tjdhfv2(args)
    # model, tokenizer = get_model_and_tokenizer_tjdllama(args)
    # model, tokenizer = get_model_and_tokenizer_nowrap(args)

    # wrap_policy = functools.partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={LlamaDecoderLayer},
    # )

    # model = FSDP(model, auto_wrap_policy=wrap_policy)

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
        # bf16=True,
        # fp16=True,  # Enable bfloat16 mixed precision
        # gradient_checkpointing=True,
        # gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        # optim="adafactor",  # Use Adafactor optimizer
        # torch_empty_cache_steps=1,
        # no_cuda=True,  # Force CPU usage
        # === FSDP ===
        # fsdp="full_shard",
        # fsdp_config={
        #     "fsdp_use_orig_params": True,
        #     "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        # },
        # fsdp_config={
        #     "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        #     "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        #     "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
        #     "fsdp_use_orig_params": True,
        #     "fsdp_cpu_ram_efficient_loading": True,
        #     # "fsdp": {
        #     #     "sharding_strategy": "FULL_SHARD",
        #     #     "auto_wrap_policy": transformer_auto_wrap_policy,
        #     #     "use_orig_params": True,
        #     #     "cpu_offload": False,
        #     #     "mixed_precision": args.bf16,
        #     # },
        #     # "activation_checkpointing": {
        #     #     "checkpointing_policy": "always",
        #     #     "contiguous_memory_optimization": True,
        #     # },
        # },
    )

    if local_rank == 0:
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

    # Log model memory breakdown
    memory_breakdown = calculate_model_memory_breakdown(
        model,
        args.batch_size,
        args.seq_len,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )

    printo("\n===== THEORETICAL MEMORY BREAKDOWN =====")
    printo(
        f"Parameters: {memory_breakdown['params_in_billions']:.2f}B parameters, {memory_breakdown['parameters_gb']:.2f} GB"
    )
    printo(f"Optimizer states: {memory_breakdown['optimizer_states_gb']:.2f} GB")
    printo(f"Activation estimate: {memory_breakdown['activations_estimate_gb']:.2f} GB")
    printo(f"Gradients: {memory_breakdown['gradients_gb']:.2f} GB")
    printo(f"Total theoretical: {memory_breakdown['total_theoretical_gb']:.2f} GB")
    printo("=========================================\n")

    # Log layer sizes
    printo("\n===== LAYER SIZE BREAKDOWN =====")
    layer_sizes = get_layer_sizes(model)
    printo("\nLayer Size Analysis:")
    for module, size in sorted(layer_sizes.items(), key=lambda x: x[1], reverse=True):
        printo(f"  {module}: {size/1e9:.2f}B parameters")
    printo("=========================================\n")

    # Initialize the trainer
    trainer = MemoryLoggingTJDTrainer(
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
            eos_token_id=int(tokenizer.eos_token_id),  # type: ignore
        ),
        on_converge_callback_cs=on_converge_callback_cs,
    )
    printr(f"Trainer initialized with {len(lm_dataset['train'])} training samples")
    trainer.train(resume_from_checkpoint=has_checkpoint)
    print(f"Experiment: {ckpt_dir})")


if __name__ == "__main__":
    main()
