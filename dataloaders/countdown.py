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
            question="Calculate 139 using all of these numbers: 36, 29, 95, 32, 4, 15.\nEach number may be used at most once.\n\nFinal answer format instructions:\n1. Provide your solution as a arithmetic expression (no '=' sign).\n2. Do not include the target number in the expression.\n3. Use '*' for multiplication.\n4. Use '/' for division.\n5. Do not include any other text or formatting.\n",
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
        ds = load_dataset(
            "mremila/countdown",
            split="train",
            cache_dir=self.cache_dir,
        )
        ds = self._process_train_dataset(ds, self.tokenizer)
        ds_dict = ds.train_test_split(test_size=200, seed=42, shuffle=True)  # type: ignore
        ds_dict["eval"] = ds_dict["test"]
        return DatasetDict(ds_dict)


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
