import random
from dataloaders.base import AbstractDataset
from datasets import DatasetDict, Dataset


class DataIterator:
    def __init__(self, num_samples, split="train", shift="in"):
        self.num_samples = num_samples
        self.split = split
        self.shift = shift

    def _clip_temp(self, temp: int) -> int:
        return max(-40, min(80, temp))

    def _sample_celsius(self):
        # Base distribution with normal shape

        if self.split == "train":
            # Normal distribution centered at 20°C
            return self._clip_temp(int(random.gauss(20, 2)))

        # Eval/Test distribution depends on shift level
        if self.shift == "in":
            return self._clip_temp(int(random.gauss(20, 2)))
        elif self.shift == "mild":
            return self._clip_temp(int(random.gauss(25, 2)))
        elif self.shift == "hard":
            return self._clip_temp(int(random.gauss(30, 2)))  # 45-35
        else:
            raise ValueError(f"Invalid shift: {self.shift}")

    def _sample_celsius_v1(self):
        # Train distribution is always the same
        if self.split == "train":
            return random.randint(-20, 40)
        # Eval/Test distribution depends on shift level
        if self.shift == "in":
            return random.randint(-20, 40)  # IID
        if self.shift == "mild":
            return random.randint(-40, 80)  # larger but overlapping
        return random.randint(80, 200)  # disjoint (hard)

    def __iter__(self):
        for _ in range(self.num_samples):
            temp_c = self._sample_celsius()
            temp_f = (temp_c * 9 / 5) + 32
            question = f"What is {temp_c}°C in Fahrenheit?"
            response = (
                "\nLet's solve this step by step:\n"
                "1) To convert Celsius to Fahrenheit, use the formula: °F = (°C x 9/5) + 32\n"
                f"2) Plugging in {temp_c}°C:\n   °F = ({temp_c} x 9/5) + 32\n   °F = {temp_f}\n\n####\n{temp_f}"
            )
            yield {"question": question, "answer": response}


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
        return (
            self.template.format(
                question=example["question"],
                answer=example["answer"],
            )
            + "\n"
            + self.eos
        )

    def format_test_example(self, example):
        return self.template.format(
            question=example["question"],
            answer="",
        )

    def format_test_label(self, example):
        return self.parse_answer(example["answer"])

    def load_raw_data(self):
        num_train_samples = 10000
        num_test_samples = 100
        base_datasets = {
            "train": Dataset.from_generator(
                lambda: DataIterator(
                    num_train_samples, split="train", shift=self.domain_shift
                )
            ),
            "eval": Dataset.from_generator(
                lambda: DataIterator(
                    num_test_samples, split="eval", shift=self.domain_shift
                )
            ),
        }
        test_datasets = {
            "test": Dataset.from_generator(
                lambda: DataIterator(
                    num_test_samples, split="test", shift=self.domain_shift
                )
            )
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
    stemp = STemp(tokenizer, max_num_samples=100).load_data()

    # Print a sample from the dataset
    print("Train example:")
    print(tokenizer.decode(stemp["train"][0]["input_ids"]))

    print("Test example (input):")
    print(tokenizer.decode(stemp["test"][0]["input_ids"]))

    print("Test example (label):")
    print(stemp["test"][0]["labels"])
