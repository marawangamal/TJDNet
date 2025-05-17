import random
from dataloaders.base import AbstractDataset
from datasets import DatasetDict, Dataset


class DataIterator:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            yield self._generate_sample()

    def _generate_sample(self):
        """Generate a single synthetic QA sample."""
        temp_c = random.randint(-20, 40)
        temp_f = (temp_c * 9 / 5) + 32

        question = f"What is {temp_c}°C in Fahrenheit?"
        response = f"\nLet's solve this step by step:\n1) To convert Celsius to Fahrenheit, use the formula: °F = (°C x 9/5) + 32\n2) Plugging in {temp_c}°C:\n   °F = ({temp_c} x 9/5) + 32\n   °F = {temp_f}\n\n####\n{temp_f}"

        return {"question": question, "answer": response}


class STemp(AbstractDataset):
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
            question="What is 20°C in Fahrenheit?",
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
        num_train_samples = self.max_num_samples if self.max_num_samples else 1000
        num_test_samples = 100
        base_datasets = {
            "train": Dataset.from_generator(lambda: DataIterator(num_train_samples)),
            "eval": Dataset.from_generator(lambda: DataIterator(num_test_samples)),
        }
        test_datasets = {
            "test": Dataset.from_generator(lambda: DataIterator(num_test_samples))
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
    stemp = STemp(tokenizer).load_data()
    print(stemp)
