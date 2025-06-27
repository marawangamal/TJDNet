from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict


class Reddit(AbstractDataset):
    """Reddit dataset dataloader for language modeling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_answer(self, generation: str) -> float:
        raise NotImplementedError(
            "Reddit dataset does not have a specific answer format."
        )

    def get_sample_prompt(self) -> str:
        return "I think that..."

    def format_train_example(self, example):
        return example["body"]

    def format_test_example(self, example):
        raise NotImplementedError(
            "Reddit dataset does not have a specific test example format."
        )

    def format_test_label(self, example):
        raise NotImplementedError(
            "Reddit dataset does not have a specific answer format."
        )

    def load_data(self):
        # Load Reddit dataset
        # Note: Reddit dataset is quite large, so we'll load a subset
        train_ds = load_dataset("reddit", split="train", trust_remote_code=True)
        validation_ds = load_dataset(
            "reddit", split="validation", trust_remote_code=True
        )
        test_ds = load_dataset("reddit", split="test", trust_remote_code=True)

        # Filter out empty or very short posts
        def filter_posts(example):
            return len(example["body"].strip()) > 50

        train_ds = train_ds.filter(filter_posts)
        validation_ds = validation_ds.filter(filter_posts)
        test_ds = test_ds.filter(filter_posts)

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
    reddit = Reddit(tokenizer, max_num_samples=100).load_data()
    print(reddit)

    # Print a sample from the dataset
    print("=" * 50)
    print("Train example:")
    print(tokenizer.decode(reddit["train"][0]["input_ids"]))
    print("-" * 50)
