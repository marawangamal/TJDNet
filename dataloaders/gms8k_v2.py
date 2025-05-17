from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict


class GSM8k(AbstractDataset):
    def __init__(self, tokenizer, seq_len=512):
        super().__init__(tokenizer, seq_len)

    def parse_answer(self, generation: str) -> float:
        try:
            return (
                float(
                    generation.split("####")[-1]
                    .split(self.eos)[0]
                    .strip()
                    .split(" ")[0]
                    .split("\n")[0]
                )
                if "####" in generation
                else float("nan")
            )
        except Exception as e:
            return float("nan")

    def format_train_example(self, example):
        return f"[QUESTION]\n{example['question']}\n[ANSWER]{example['answer']}"

    def format_test_example(self, example):
        return f"[QUESTION]\n{example['question']}\n[ANSWER]"

    def format_test_label(self, example):
        return self.parse_answer(example["answer"])

    def load_data(self):
        base_datasets = {
            "train": load_dataset("openai/gsm8k", "main", split="train"),
            "val": load_dataset("openai/gsm8k", "main", split="test"),
        }
        test_datasets = {
            "test": load_dataset("openai/gsm8k", "main", split="test"),
        }

        for split in base_datasets:
            base_datasets[split] = self._process_train_dataset(
                base_datasets[split], self.tokenizer
            )
        for split in test_datasets:
            test_datasets[split] = self._process_test_dataset(
                test_datasets[split], self.tokenizer
            )

        return DatasetDict({**base_datasets, **test_datasets})


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gsm8k = GSM8k(tokenizer)
    print(gsm8k.dataset)
