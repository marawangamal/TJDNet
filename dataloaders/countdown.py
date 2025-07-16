from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict, Dataset
import random


class Countdown(AbstractDataset):
    templates = {
        "0_shot": """[QUESTION]\n{question}\n[ANSWER]{answer}""",
        "few_shot": (
            "You are a helpful assistant that solves countdown problems. "
            "The final answer should be preceeded by ####.\n"
            "Here is an example:\n"
            "[QUESTION]\n"
            "Count down from 10 to 1. What is the 3rd number in this sequence?\n"
            "[ANSWER]\n"
            "Counting down from 10: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1\n"
            "The 3rd number in this countdown sequence is 8. #### 8\n"
            "Now, answer the following question:\n"
            "[QUESTION]\n"
            "{question}\n"
            "[ANSWER]\n"
            "{answer}"
        ),
        "few_shot:standard": (
            "You are a helpful assistant that solves countdown problems. "
            "The final answer should be preceeded by ####.\n"
            "Here is an example:\n"
            "[QUESTION]\n"
            "Count down from 10 to 1. What is the 3rd number in this sequence?\n"
            "[ANSWER]\n"
            "The 3rd number in the countdown sequence is 8. #### 8\n"
            "Now, answer the following question:\n"
            "[QUESTION]\n"
            "{question}\n"
            "[ANSWER]\n"
            "{answer}"
        ),
    }

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

    def get_sample_prompt(self) -> str:
        return self.templates[self.template_mode].format(
            question="Count down from 15 to 1. What is the 5th number in this sequence?",
            answer="",
        )

    def format_train_example(self, example):
        return (
            self.templates[self.template_mode].format(
                question=example["question"],
                answer=example["answer"],
            )
            + self.eos
        )

    def format_test_example(self, example):
        return self.templates[self.template_mode].format(
            question=example["question"],
            answer="",
        )

    def format_test_label(self, example):
        return self.parse_answer(example["answer"])

    def _generate_countdown_problem(self, start_num, target_position):
        """Generate a countdown problem with given start number and target position."""
        if target_position > start_num:
            return None, None

        # Generate the countdown sequence
        sequence = list(range(start_num, 0, -1))
        target_number = sequence[target_position - 1]  # Convert to 0-indexed

        question = f"Count down from {start_num} to 1. What is the {target_position}{self._get_ordinal_suffix(target_position)} number in this sequence?"
        answer = f"Counting down from {start_num}: {', '.join(map(str, sequence))}\nThe {target_position}{self._get_ordinal_suffix(target_position)} number in this countdown sequence is {target_number}. #### {target_number}"

        return question, answer

    def _get_ordinal_suffix(self, n):
        """Get the ordinal suffix for a number."""
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return suffix

    def _generate_dataset_examples(self, num_examples, split="train"):
        """Generate synthetic countdown problems."""
        for i in range(num_examples):
            # Vary difficulty based on split
            if split == "train":
                start_num = random.randint(5, 50)
                target_position = random.randint(1, min(start_num, 10))
            elif split == "eval":
                start_num = random.randint(10, 30)
                target_position = random.randint(1, min(start_num, 8))
            else:  # test
                start_num = random.randint(15, 40)
                target_position = random.randint(1, min(start_num, 12))

            question, answer = self._generate_countdown_problem(
                start_num, target_position
            )
            if question and answer:
                yield {
                    "question": question,
                    "answer": answer,
                    "start_num": start_num,
                    "target_position": target_position,
                    "target_number": start_num - target_position + 1,
                }

    def load_raw_data(self):
        # Generate synthetic countdown problems since reasoning-gym dataset might not be available
        num_train = 5000
        num_eval = 500
        num_test = 500

        # Create datasets using Dataset.from_generator pattern like STemp
        base_datasets = {
            "train": Dataset.from_generator(
                lambda: self._generate_dataset_examples(num_train, "train")
            ),
            "eval": Dataset.from_generator(
                lambda: self._generate_dataset_examples(num_eval, "eval")
            ),
        }
        test_datasets = {
            "test": Dataset.from_generator(
                lambda: self._generate_dataset_examples(num_test, "test")
            ),
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
    tokenizer.pad_token = tokenizer.eos_token

    countdown = Countdown(tokenizer, max_num_samples=100).load_data()
    print(countdown)

    # Print a sample from the dataset
    print("Train example:")
    print(tokenizer.decode(countdown["train"][0]["input_ids"]))

    print("Test example:")
    print(tokenizer.decode(countdown["test"][0]["input_ids"]))
