import os
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
        return example["text"]

    def format_test_example(self, example):
        raise NotImplementedError(
            "Shakespeare dataset does not have a specific test example format."
        )

    def format_test_label(self, example):
        raise NotImplementedError(
            "Shakespeare dataset does not have a specific answer format."
        )

    def load_data(self):
        local_path = os.path.join("data", "tinyshakespeare.txt")
        ds = load_dataset("text", data_files={"train": local_path}, split="train")
        ds_dict = ds.train_test_split(test_size=0.1, shuffle=False)
        ds_dict["eval"] = ds_dict["test"]

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
