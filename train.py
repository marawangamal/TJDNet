# python train.py --model_type llama --model_head base --horizon 1 --horizon_eval 1 --dataset sharegpt --freeze_base_model --batch_size 2 --seq_len 32
import os.path as osp
import os
import wandb

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from data.shakespeare import load_shakespeare_data
from data.sharegpt import load_sharegpt_data
from data.wikitext import load_wikitext_data
from utils import get_experiment_name
from helpers import (
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
    ckpt_dir = osp.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)
    save_args(args, ckpt_dir)

    # Model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)

    # Datasets
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
        "sharegpt": load_sharegpt_data,
    }[args.dataset](tokenizer, args.seq_len)
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
        logging_strategy="steps",  # Changed from "steps" to "epoch"
        logging_steps=100,
        logging_first_step=True,
        # Evaluation
        eval_strategy="epoch",  # Evaluate every epoch
        # Reporting
        report_to="wandb",  # Enable wandb logging
        # Checkpoints
        save_strategy="best",  # Save model every epoch
        save_safetensors=False,
        save_total_limit=1,
        metric_for_best_model="eval_nll",
        greater_is_better=False,
    )

    # Initialize wandb only on main process
    if training_args.local_rank == 0:  # main process
        wandb.init(
            project="tjdnet-sharegpt-dev",
            name=exp_name,
        )

    # Initialize the trainer
    trainer = TJDTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

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
