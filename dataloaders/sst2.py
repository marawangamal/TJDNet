from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict


class SST2(AbstractDataset):
    """SST-2 (Stanford Sentiment Treebank) dataset dataloader for sentiment analysis."""

    templates = {
        "0_shot": "Sentence: {sentence}\nSentiment: {sentiment}",
        "few_shot": (
            "You are a helpful assistant that analyzes sentiment. "
            "The sentiment should be either 'positive' or 'negative'.\n"
            "Here is an example:\n"
            "Sentence: This movie is absolutely fantastic!\n"
            "Sentiment: positive\n"
            "Now, analyze the following sentence:\n"
            "Sentence: {sentence}\n"
            "Sentiment: {sentiment}"
        ),
        "few_shot:standard": (
            "You are a helpful assistant that analyzes sentiment. "
            "The sentiment should be either 'positive' or 'negative'.\n"
            "Here is an example:\n"
            "Sentence: This movie is absolutely fantastic!\n"
            "Sentiment: positive\n"
            "Now, analyze the following sentence:\n"
            "Sentence: {sentence}\n"
            "Sentiment: {sentiment}"
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_map = {0: "negative", 1: "positive"}

    def parse_answer(self, generation: str) -> float:
        try:
            if "####" in generation:
                answer = (
                    generation.split("####")[-1]
                    .split(self.eos)[0]
                    .strip()
                    .split(" ")[0]
                    .split("\n")[0]
                )
            else:
                # Try to extract sentiment from the last line
                lines = generation.strip().split("\n")
                answer = lines[-1].strip().lower()

            # Map sentiment to numeric label
            if "positive" in answer:
                return 1.0
            elif "negative" in answer:
                return 0.0
            else:
                return float("nan")
        except Exception as e:
            return float("nan")

    def get_sample_prompt(self) -> str:
        return self.templates[self.template_mode].format(
            sentence="This movie is absolutely fantastic!",
            sentiment="",
        )

    def format_train_example(self, example):
        sentiment = self.label_map[example["label"]]
        return self.templates[self.template_mode].format(
            sentence=example["sentence"],
            sentiment=f"{sentiment} {self.eos}",
        )

    def format_test_example(self, example):
        return self.templates[self.template_mode].format(
            sentence=example["sentence"],
            sentiment="",
        )

    def format_test_label(self, example):
        # Test set has labels of -1 (no label available), so we'll use 0 as default
        # but this should be handled appropriately in evaluation
        label = example["label"]
        if label == -1:
            return 0.0  # Default value for test set
        return float(label)

    def load_data(self):
        # Load SST-2 dataset
        train_ds = load_dataset("glue", "sst2", split="train")
        validation_ds = load_dataset("glue", "sst2", split="validation")
        test_ds = load_dataset("glue", "sst2", split="test")

        # Process datasets
        train_ds = self._process_train_dataset(train_ds, self.tokenizer)
        validation_ds = self._process_test_dataset(validation_ds, self.tokenizer)
        test_ds = self._process_test_dataset(test_ds, self.tokenizer)

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
    sst2 = SST2(tokenizer, max_num_samples=100).load_data()
    print(sst2)

    # Print a sample from the dataset
    print("=" * 50)
    print("Train example:")
    print(tokenizer.decode(sst2["train"][0]["input_ids"]))
    print("-" * 50)
    print("Test example (input_ids):")
    print(tokenizer.decode(sst2["test"][0]["input_ids"]))
    print("Test example (labels):")
    print(sst2["test"][0]["labels"])
    print("-" * 50)
