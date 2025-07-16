from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict, Dataset
import reasoning_gym


class Countdown(AbstractDataset):
    templates = {
        "0_shot": """[QUESTION]\n{question}\n[ANSWER]{answer}""",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_answer(self, generation: str) -> float:
        try:
            result = 0.0
            curr_op = "+"  # +, -, *, /
            answer_arr = generation.split(" ")
            for i, num in enumerate(answer_arr):
                if num.isdigit():
                    # result += int(num) * (10 ** (len(answer_arr) - i - 1))
                    if curr_op == "+":
                        result += int(num)
                    elif curr_op == "-":
                        result -= int(num)
                    elif curr_op == "*":
                        result *= int(num)
                    elif curr_op == "/":
                        result /= int(num)
                    else:
                        raise ValueError(f"Invalid operator: {curr_op}")
                elif num in ["+", "-", "*", "/"]:
                    curr_op = num
                else:
                    raise ValueError(f"Invalid number: {num}")
            return result
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
        return float(self.parse_answer(example["metadata"]["target"]))

    def load_raw_data(self):
        # Generate synthetic countdown problems since reasoning-gym dataset might not be available
        train_ds = reasoning_gym.create_dataset("countdown", size=1000, seed=42)
        eval_ds = reasoning_gym.create_dataset("countdown", size=100, seed=42)
        test_ds = reasoning_gym.create_dataset("countdown", size=100, seed=42)

        # Convert to huggingface dataset
        train_ds = Dataset.from_list(train_ds)
        eval_ds = Dataset.from_list(eval_ds)
        test_ds = Dataset.from_list(test_ds)

        # Process the datasets
        train_ds = self._process_train_dataset(train_ds, self.tokenizer)
        eval_ds = self._process_train_dataset(eval_ds, self.tokenizer)
        test_ds = self._process_test_dataset(test_ds, self.tokenizer)
        return DatasetDict(
            **{
                "train": train_ds,
                "eval": eval_ds,
                "test": test_ds,
            }
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    countdown = Countdown(tokenizer, max_num_samples=100).load_data()
    print(countdown)

    # Print a sample from the dataset
    print("Train example:")
    print(tokenizer.decode(countdown["train"][0]["input_ids"]))

    print("\n\n\nTest example:")
    print(tokenizer.decode(countdown["test"][0]["input_ids"]))
