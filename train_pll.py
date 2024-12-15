"""
Fine-tune GPT-2 using TJDNet on Shakespeare dataset.

Example usage:
python train_clm_shakespeare_char_tjd.py --model cpgpt2 

Resources: 
https://huggingface.co/docs/transformers/tasks/language_modeling
https://github.com/dariush-bahrami/character-tokenizer/tree/master


Two options for data loading:

Given a dataset of sequences of different length {s1, s2, ..., s2}, we have two options for dataloading

1. Simple (preprocess_simple)
    - Convert each sequence to be of length `max_len` via padding or trunction 

2. Advanced (preprocess_function & group texts)
    - Combine to sinlge length string s = [s_1, s_2, ..., s_b], then split into chunks of size `max_len`. This is less 
    - Less wastefulness from truncation


"""

from typing import Dict, Any
import os.path as osp
import os
import wandb

from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

from helpers import parse_args, set_seed
from models.tjdgpt2 import TJDGPT2
from ctokenizers.char_tokenizer import CharTokenizer
from data.shakespeare import load_shakespeare_data
from data.wikitext import load_wikitext_data
from utils import get_experiment_name


def get_test_samples(
    model,
    tokenizer,
    prompt="\n",
    max_new_tokens=8,
    top_k=200,
    num_beams=5,
    do_sample=True,
    horizon_eval=1,
    n_samples=1,
    print_output=True,
):
    # Inference
    model.eval()
    samples = []
    for i in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            inputs,
            num_beams=num_beams,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            horizon=horizon_eval,
            top_k=top_k,
        )
        sample = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if n_samples == 1:
            samples.append(sample)
        else:
            samples.append(f"[{i+1}] {sample}")

    if print_output:
        print("\n---\n".join(samples) + "\n")
    return "\n".join(samples)


class TJDTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        output_dict = model(**inputs)
        loss = output_dict["loss"]
        return (loss, output_dict) if return_outputs else loss


def main():
    args = parse_args()
    set_seed(args.seed)

    # Configuration
    exp_name = get_experiment_name(vars(args))
    ckpt_dir = osp.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Tokenizer
    tokenizer = (
        AutoTokenizer.from_pretrained("gpt2")
        if args.tokenizer_type == "word"
        else CharTokenizer(args.seq_len)
    )
    if args.tokenizer_type == "word":
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
    }[args.dataset](tokenizer, args.seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Model configuration
    model_config = {
        "model_head": args.model_head,
        "vocab_size": (
            len(tokenizer.get_vocab())
            if hasattr(tokenizer, "get_vocab")
            else len(tokenizer)
        ),
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "dropout": args.dropout,
        "rank": args.rank,
        "horizon": args.horizon,
        "positivity_func": args.positivity_func,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "freeze_base_model": args.freeze_base_model,
    }
    model = TJDGPT2(**model_config)
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
        logging_strategy="epoch",  # Changed from "steps" to "epoch"
        logging_first_step=True,
        # Checkpoints
        save_strategy="no",  # Disable saving
        # Evaluation
        eval_strategy="epoch",  # Evaluate every epoch
        # Reporting
        report_to="wandb",  # Enable wandb logging
    )

    # Initialize wandb only on main process
    if training_args.local_rank == 0:  # main process
        wandb.init(
            project="tjdnet-shakepeare-dev",
            name=exp_name,
        )

    # Custom evaluation function
    def compute_metrics(eval_pred):
        # Note: If return type of model forward is a dict, then the `predictions` will be tuple of all vals of keys except loss
        # See `prediction_step` in Trainer class
        (nll, loss_scale), labels = eval_pred
        return {
            "nll": nll.mean().item(),
        }

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
    test_sample = get_test_samples(model, tokenizer, max_new_tokens=args.max_new_tokens)
    print(f"Test sample:\n{test_sample}")


if __name__ == "__main__":
    main()
