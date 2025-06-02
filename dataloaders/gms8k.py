from networkx import d_separated
from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict


class GSM8k(AbstractDataset):
    template = """[QUESTION]\n{question}\n[ANSWER]{answer}"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    @classmethod
    def get_sample_prompt(cls) -> str:
        return cls.template.format(
            question="Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            answer="",
        )

    def format_train_example(self, example):
        return self.template.format(
            question=example["question"],
            answer=example["answer"],
        )

    def format_test_example(self, example):
        return self.template.format(
            question=example["question"],
            answer="",
        )

    def format_test_label(self, example):
        return self.parse_answer(example["answer"])

    def load_data(self):

        base_datasets = {
            "train": load_dataset("openai/gsm8k", "main", split="train"),
            "eval": load_dataset("openai/gsm8k", "main", split="test"),
        }
        test_datasets = {
            "test": load_dataset("openai/gsm8k", "main", split="test"),
        }
        # AdHoc limit test set to 50% of total
        test_datasets["test"] = (
            test_datasets["test"]
            .shuffle(seed=42)
            .select(range(int(len(test_datasets["test"]) * 0.5)))
        )

        for split in base_datasets:
            base_datasets[split] = self._process_train_dataset(
                base_datasets[split], self.tokenizer
            )
        for split in test_datasets:
            test_datasets[split] = self._process_test_dataset(
                test_datasets[split], self.tokenizer
            )

        ds = DatasetDict({**base_datasets, **test_datasets})

        # Limit the number of samples to 1000 for train and 100 for eval/test
        if self.max_num_samples is not None:
            for split in ds:
                ds[split] = (
                    ds[split].shuffle(seed=42).select(range(self.max_num_samples))
                )

        return ds


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gsm8k = GSM8k(tokenizer, max_num_samples=100).load_data()
    print(gsm8k)
