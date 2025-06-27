from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict


class WikiText2(AbstractDataset):
    """WikiText-2 dataset dataloader for language modeling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_answer(self, generation: str) -> float:
        raise NotImplementedError(
            "WikiText-2 dataset does not have a specific answer format."
        )

    def get_sample_prompt(self) -> str:
        return "The quick brown fox jumps over the lazy dog."

    def format_train_example(self, example):
        return example["text"]

    def format_test_example(self, example):
        raise NotImplementedError(
            "WikiText-2 dataset does not have a specific test example format."
        )

    def format_test_label(self, example):
        raise NotImplementedError(
            "WikiText-2 dataset does not have a specific answer format."
        )

    def load_data(self):
        # Load WikiText-2 dataset
        train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        validation_ds = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="validation"
        )
        test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        # Process datasets
        train_ds = self._process_train_dataset(train_ds, self.tokenizer)
        validation_ds = self._process_train_dataset(validation_ds, self.tokenizer)
        test_ds = self._process_train_dataset(test_ds, self.tokenizer)

        ds_dict = {
            "train": train_ds,
            "eval": validation_ds,
            "test": test_ds,
        }

        # Apply token limits first, then sample limits
        if self.max_tokens is not None:
            for split in ds_dict:
                ds_dict[split] = self._limit_by_tokens(ds_dict[split], split)
        elif self.max_num_samples is not None:
            for split in ds_dict:
                if len(ds_dict[split]) > self.max_num_samples:
                    ds_dict[split] = (
                        ds_dict[split]
                        .shuffle(seed=42)
                        .select(range(self.max_num_samples))
                    )

        return DatasetDict(ds_dict)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    wikitext2 = WikiText2(tokenizer, max_num_samples=100).load_data()
    print(wikitext2)

    # Print a sample from the dataset
    print("=" * 50)
    print("Train example:")
    print(tokenizer.decode(wikitext2["train"][0]["input_ids"]))
    print("-" * 50)
