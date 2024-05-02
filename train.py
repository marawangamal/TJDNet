from tqdm.auto import tqdm
import argparse
import logging
import torch.nn as nn

from transformers import (
    get_scheduler,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, formatting

from torch.utils.data import DataLoader
import torch
from TJDNet import TJDNet, TJDLayer

logging.basicConfig(
    format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def preprocess_wikitext_batch(examples: formatting.formatting.LazyBatch):
    return examples["text"]


def preprocess_e2e_nlg_batch(examples: formatting.formatting.LazyBatch):
    in_out_pairs = zip(examples["meaning_representation"], examples["human_reference"])  # type: ignore
    output = ["Input: {} Output: {}".format(x, y) for x, y in in_out_pairs]
    return output


DATASET_CONFIGS = {
    "wikitext": {
        "subset": "wikitext-103-v1",
        "train_split": "train",
        "eval_split": "test",
        "preprocess_batch_func": preprocess_wikitext_batch,
        "eval_prompt": "The meaning of life is",
    },
    "e2e_nlg": {
        "subset": None,
        "train_split": "train",
        "eval_split": "test",
        "preprocess_batch_func": preprocess_e2e_nlg_batch,
        "eval_prompt": "Input: name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], near[Café Adriatic] Output: ",
    },
}


TJD_MODEL_CONFIG = {
    "gpt2": {
        "head_name": "lm_head",
        "emb_size": 768,
        "vocab_size": 50257,
        "condition_func": lambda lyr, lyr_name: lyr_name == "lm_head",
        "replacement_func": lambda lyr: TJDLayer(
            emb_size=768, rank=2, vocab_size=50257
        ),
    }
}


def train(
    model_name: str,
    dataset_name: str,
    debug: bool = False,
    max_seq_len: int = 4,
    num_epochs: int = 3,
    batch_size: int = 4,
):
    # Load the dataset
    logger.info(
        f"Training model {model_name} on dataset {dataset_name} in {'debug' if debug else 'full'} mode"
    )
    config = DATASET_CONFIGS[dataset_name]
    tjd_config = TJD_MODEL_CONFIG[model_name]
    dataset = (
        load_dataset(dataset_name, config["subset"])
        if config["subset"]
        else load_dataset(dataset_name)
    )

    if debug:
        for split in dataset.keys():  # type: ignore
            dataset[split] = dataset[split].select(range(50))  # type: ignore

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = TJDNet(
        model,
        condition_func=tjd_config["condition_func"],
        replacement_func=tjd_config["replacement_func"],
    )
    model.replace_base_model_layers()

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        logger.info(
            "Tokenizer does not have a pad token set. Setting pad_token to eos_token."
        )
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess the dataset
    def tokenize_function(examples, max_length=max_seq_len):
        preprocessed = config["preprocess_batch_func"](examples)
        result = tokenizer(
            preprocessed,
            # padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return result

    tokenized_dataset_train = dataset[config["train_split"]].map(tokenize_function, batched=True)  # type: ignore
    tokenized_dataset_train = tokenized_dataset_train.remove_columns(dataset[config["train_split"]].column_names)  # type: ignore
    tokenized_dataset_eval = dataset[config["eval_split"]].map(tokenize_function, batched=True)  # type: ignore
    tokenized_dataset_eval = tokenized_dataset_eval.remove_columns(dataset[config["eval_split"]].column_names)  # type: ignore

    # Language Modeling Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_dataloader = DataLoader(
        tokenized_dataset_train,  # type: ignore
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        tokenized_dataset_eval, batch_size=batch_size, collate_fn=data_collator  # type: ignore
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )  # type: ignore

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

            log_str = f"Epoch {epoch + 1} | Step {progress_bar.n}/{num_training_steps} | Loss: {loss.item():.4f}"
            progress_bar.set_description_str(log_str)

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

        generate_text_sample(model, tokenizer, device, prompt=config["eval_prompt"])  # type: ignore


def generate_text_sample(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompt: str = "The meaning of life is",
):
    logger.info("Generating text sample...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # type: ignore
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

    parser.add_argument(
        "--max_seq_len", type=int, default=4, help="Maximum sequence length"
    )

    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        debug=args.debug,
        max_seq_len=args.max_seq_len,
        num_epochs=args.num_epochs,
    )
