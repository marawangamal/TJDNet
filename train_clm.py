from tqdm.auto import tqdm
import argparse
import logging

from transformers import (
    get_scheduler,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

from torch.utils.data import DataLoader
import torch
from TJDNet import TJDNet
from TJDNet import RepNet

logging.basicConfig(
    format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


DATASET_CONFIGS = {
    "wikitext": {
        "subset": "wikitext-103-v1",
        "train_split": "train",
        "test_split": "test",
    }
}


def train(model_name: str, dataset_name: str, debug: bool = False):
    # Load the dataset
    logger.info(
        f"Training model {model_name} on dataset {dataset_name} in {'debug' if debug else 'full'} mode"
    )
    config = DATASET_CONFIGS[dataset_name]
    dataset = load_dataset(dataset_name, config["subset"])

    if debug:
        for split in dataset.keys():  # type: ignore
            dataset[split] = dataset[split].select(range(500))  # type: ignore

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = TJDNet(model, emb_size=128)
    # model = RepNet(model, condition_func=lambda x, y: False, replacement_func=lambda x: TJDNet(emb_size=128))  # type: ignore
    model.replace_base_model_layers()

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        logger.info(
            "Tokenizer does not have a pad token set. Setting pad_token to eos_token."
        )
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    tokenized_dataset_train = dataset[config["train_split"]].map(tokenize_function, batched=True)  # type: ignore
    tokenized_dataset_train = tokenized_dataset_train.remove_columns(dataset[config["train_split"]].column_names)  # type: ignore
    tokenized_dataset_eval = dataset[config["test_split"]].map(tokenize_function, batched=True)  # type: ignore
    tokenized_dataset_eval = tokenized_dataset_eval.remove_columns(dataset[config["test_split"]].column_names)  # type: ignore

    # Language Modeling Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_dataloader = DataLoader(
        tokenized_dataset_train,  # type: ignore
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        tokenized_dataset_eval, batch_size=8, collate_fn=data_collator  # type: ignore
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        avg_train_loss = train_loss / len(train_dataloader)

        # Evaluation loop
        model.eval()
        total_eval_loss = 0
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            total_eval_loss += outputs.loss.item()
        avg_eval_loss = total_eval_loss / len(eval_dataloader)

        logger.info(
            f"Epoch {epoch + 1} | Training Loss: {avg_train_loss:.4f} | Evaluation Loss: {avg_eval_loss:.4f}"
        )

        generate_text_sample(model, tokenizer, device)


def generate_text_sample(model, tokenizer, device):
    logger.info("Generating text sample...")
    input_ids = tokenizer.encode("The meaning of life is", return_tensors="pt").to(
        device
    )
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text:\n{generated_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    train(args.model_name, args.dataset_name, args.debug)
