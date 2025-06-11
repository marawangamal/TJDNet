from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict


class Shakespeare(AbstractDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_answer(self, generation: str) -> float:
        raise NotImplementedError(
            "Shakespeare dataset does not have a specific answer format."
        )

    def get_sample_prompt(self) -> str:
        return "KATHARINA:"

    def format_train_example(self, example):
        verses = example["text"].split("\n\n")
        return f"{self.eos}".join(verses) + self.eos

    def format_test_example(self, example):
        raise NotImplementedError(
            "Shakespeare dataset does not have a specific test example format."
        )

    def format_test_label(self, example):
        raise NotImplementedError(
            "Shakespeare dataset does not have a specific answer format."
        )

    def load_data(self):
        ds = load_dataset(
            "karpathy/tiny_shakespeare",
            split=(
                f"train[:{self.max_num_samples}]" if self.max_num_samples else "train"
            ),
            # use local
            data_files={"train": "tinyshakespeare.txt"},
        )
        ds_dict = {
            "train": ds,
            "eval": ds,
            "test": ds,
        }

        for split in ds_dict:
            ds_dict[split] = self._process_train_dataset(ds_dict[split], self.tokenizer)

        return DatasetDict(ds_dict)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    speare = Shakespeare(tokenizer, max_num_samples=100).load_data()
    # Print a sample from the dataset
    print("Train example:")
    print(tokenizer.decode(speare["train"][0]["input_ids"]))
