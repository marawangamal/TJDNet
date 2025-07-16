from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict, Dataset


class FineWeb(AbstractDataset):
    """Dataloader for the HuggingFaceTB/smollm-corpus fineweb-edu-dedup subset (small sample)."""

    template = "<|user|> {prompt}\n<|assistant|> {response}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_sample_prompt(cls) -> str:
        return cls.template.format(prompt="What is SmolLM?", response="")

    def parse_answer(self, generation: str):
        # Not used for this dataset
        return ""

    def format_train_example(self, example):
        # Use the 'text' field directly
        return example["text"] + self.eos

    def format_test_example(self, example):
        raise NotImplementedError("SmolLM corpus does not have a test set.")

    def format_test_label(self, example):
        raise NotImplementedError("SmolLM corpus does not have a test set.")

    def load_raw_data(self):
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            cache_dir=self.cache_dir,
        )
        # Only select a small subset if ds is a Dataset (not IterableDataset)
        ds = self._process_train_dataset(ds, self.tokenizer)
        ds_dict = ds.train_test_split(test_size=0.1, seed=42, shuffle=True)
        ds_dict["eval"] = ds_dict["test"]
        return DatasetDict(ds_dict)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    smol = FineWeb(tokenizer, max_num_samples=100).load_data()
    print(smol)
    print("Train example:")
    print(tokenizer.decode(smol["train"][0]["input_ids"]))
    print("Eval example:")
    print(tokenizer.decode(smol["eval"][0]["input_ids"]))
