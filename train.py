# from typing import Any, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import argparse


DATASET_CONFIGS = {
    "wikitext": {
        "subset": "wikitext-103-v1",
        "train_split": "train",
        "test_split": "test",
    }
}


def train(model_name: str, dataset_name: str, task: str, debug: bool = False):
    # Load the dataset
    config = DATASET_CONFIGS[dataset_name]
    dataset = load_dataset(dataset_name, config["subset"])

    if debug:
        for split in dataset.keys():  # type: ignore
            dataset[split] = dataset[split].select(range(500))  # type: ignore

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        print(
            "Tokenizer does not have a pad token set. Setting pad_token to eos_token."
        )
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Language Modeling Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_dataset = tokenized_datasets[config["train_split"]]  # type: ignore
    eval_dataset = tokenized_datasets[config["test_split"]]  # type: ignore

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_strategy="steps",
        report_to=["tensorboard"],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on a specified dataset for a specified task using Hugging Face Transformers."
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="Model name or path"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="wikitext", help="Dataset name"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use a smaller subset of the dataset for debugging",
    )

    args = parser.parse_args()

    train(args.model_name, args.dataset_name, "language-modeling", args.debug)
