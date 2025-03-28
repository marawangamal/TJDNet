"""Training script for TJDNet models.

This script trains and evaluates TJDNet models using the Hugging Face Transformers library.

Example:
    accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py \
        --epochs 20 \
        --batch_size 32 \
        --seq_len 128 \
        --dataset gsm8k \
        --model_type llama7b \
        --lr 1e-5 \
        --model_head cp \
        --num_layers 2 \
        --hidden_dim 768 \
        --horizon 2 \
        --horizon_eval 2 \
        --rank 2

Hardware requirements:
    - LLAMA 7B model: 4x GPUs w/ 80GB VRAM (FSDP)
    - GPT-2 model: 1x GPUs w/ 40GB VRAM

"""

import json
import os
import os.path as osp
from re import L
import time
import uuid
import torch.distributed as dist

import wandb
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


from utils.accuracy import compute_accuracy
from utils.generation import GenerationCallback
from dataloaders.gsm8k import load_gsm8k_data
from dataloaders.shakespeare import load_shakespeare_data
from dataloaders.sharegpt import load_sharegpt
from dataloaders.syn_number_bases import load_syn_num_base_data
from dataloaders.syn_numbers import load_syn_num_data
from dataloaders.syn_temp import load_syn_temp_data
from dataloaders.wikitext import load_wikitext_data
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

CHECKPOINT_DIR = "checkpoints"


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
        if self.test_dataset:
            acc, _ = compute_accuracy(
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
        has_checkpoint = len(checkpoint_files) > 0
        if has_checkpoint:
            print(f"Resuming from checkpoint: {ckpt_dir}")

    return args, exp_name, ckpt_dir, has_checkpoint


def main():
    # Configuration
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args = parse_args()
    args, exp_name, ckpt_dir, has_checkpoint = setup(args, local_rank)
    set_seed(args.seed)

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
        "sharegpt": load_sharegpt,
        "gsm8k": load_gsm8k_data,
        "stemp": load_syn_temp_data,
        "snum": load_syn_num_data,
        "sbase": load_syn_num_base_data,
    }[args.dataset](
        tokenizer,
        args.seq_len,
        max_num_samples=args.max_num_samples,
        print_stats=local_rank == 0,
    )
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
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps,
        save_total_limit=2,  # Save only 3 checkpoints
        load_best_model_at_end=True,
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
            config={**vars(args), **git_info},
            resume="allow",
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


if __name__ == "__main__":
    main()
