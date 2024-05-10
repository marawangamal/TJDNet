import argparse
import logging
import os.path as osp
import shutil

from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, formatting
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from TJDNet import TJDLayer
from littjdnet import LitTJDNet
from utils.utils import get_experiment_name

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
            emb_size=768, rank=2, vocab_size=50257, mode="tjd"  # ["tjd", "lm"]
        ),
    }
}


def main(
    model_name: str,
    dataset_name: str,
    lr: float = 5e-5,
    seq_len: int = 4,
    epochs: int = 20,
    batch_size: int = 4,
    checkpoint_dir: str = "checkpoints",
    overwrite: bool = True,
):

    # 0. Create a unique experiment name
    experiment_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "lr": lr,
        "model_name": model_name,
        "dataset_name": dataset_name,
    }
    experiment_name = get_experiment_name(experiment_config)
    logger.info(f"Experiment configuration\n: {experiment_config}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}/{experiment_name}")

    if osp.exists(osp.join(checkpoint_dir, experiment_name)) and overwrite:
        logger.info("Overwriting existing checkpoints...")
        shutil.rmtree(osp.join(checkpoint_dir, experiment_name))

    # 1. Load data
    config = DATASET_CONFIGS[dataset_name]
    tjd_config = TJD_MODEL_CONFIG[model_name]
    dataset = (
        load_dataset(dataset_name, config["subset"])
        if config["subset"]
        else load_dataset(dataset_name)
    )

    model_params = {
        "condition_func": tjd_config["condition_func"],
        "replacement_func": tjd_config["replacement_func"],
    }

    lit_model = LitTJDNet(model_params=model_params, model_name=model_name, lr=lr)
    tokenizer = lit_model.tokenizer

    # Preprocess the dataset
    def tokenize_function(examples, max_length=seq_len):
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

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Monitor validation loss to determine the best model
        mode="min",  # `min` mode will save the model with the lowest val_loss
        dirpath=osp.join(checkpoint_dir, experiment_name),
        filename="best",
        save_top_k=1,
        verbose=True,
        enable_version_counter=False,
        save_last=True,  # Additionally, save the most recent checkpoint after each epoch
    )
    tb_logger = TensorBoardLogger(
        osp.join(checkpoint_dir, experiment_name), name="", version=""
    )
    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        overfit_batches=1,
        log_every_n_steps=1,
        logger=tb_logger,
        gradient_clip_val=1.0,
    )
    trainer.fit(lit_model, train_dataloader, eval_dataloader, ckpt_path="last")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="Model name or path"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="wikitext", help="Dataset name"
    )

    parser.add_argument(
        "--max_seq_len", type=int, default=2, help="Maximum sequence length"
    )

    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )

    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        seq_len=args.max_seq_len,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
