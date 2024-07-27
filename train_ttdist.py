from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

import argparse
import wandb

from utils.utils import get_experiment_name


def preprocess_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=max_length
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer model on the Penn Treebank dataset at character level."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="maximum length of the input sequence (default: 512)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./data",
        help="root directory for storing the dataset (default: ./data)",
    )

    args = parser.parse_args()

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Preprocess the dataset
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length), batched=True
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize WandB
    wandb.init(
        project="tjdnet-transformer",
        config=vars(args),
        name=get_experiment_name(vars(args)),
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # type: ignore
        eval_dataset=tokenized_datasets["validation"],  # type: ignore
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    main()
