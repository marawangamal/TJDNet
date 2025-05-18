from abc import ABC, abstractmethod
from typing import Optional, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import DatasetDict


class AbstractDataset(ABC):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        seq_len: int = 512,
        max_num_samples: Optional[int] = None,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos: str = tokenizer.eos_token  # type: ignore
        self.max_num_samples = max_num_samples

    @classmethod
    @abstractmethod
    def get_sample_prompt(cls) -> str:
        """Get a sample prompt for the dataset"""
        pass

    @abstractmethod
    def parse_answer(self, generation: str) -> float:
        """Parse the answer from the generated text"""
        pass

    @abstractmethod
    def load_data(self) -> DatasetDict:
        """Load the dataset"""
        pass

    @abstractmethod
    def format_train_example(self, example) -> str:
        """Format the training example"""
        pass

    @abstractmethod
    def format_test_example(self, example) -> str:
        """Format the test example"""
        pass

    @abstractmethod
    def format_test_label(self, example) -> float:
        """Format the test label"""
        pass

    @staticmethod
    def group_texts(examples, input_seq_len):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])  # type: ignore
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= input_seq_len:
            total_length = (total_length // input_seq_len) * input_seq_len
        # Split by chunks of input_seq_len.
        result = {
            k: [t[i : i + input_seq_len] for i in range(0, total_length, input_seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def _process_train_dataset(self, dataset, tokenizer):
        # Format text using template
        dataset = dataset.map(
            lambda x: {"text": self.format_train_example(x) + self.eos},
            remove_columns=dataset.column_names,
        )

        # Tokenize
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], add_special_tokens=False),
            remove_columns=["text"],
        )

        # Group instead of padding/truncation
        dataset = dataset.map(lambda x: self.group_texts(x, self.seq_len), batched=True)
        return dataset

    def _process_test_dataset(self, dataset, tokenizer):
        # Format text using template
        dataset = dataset.map(
            lambda x: {
                "text": self.format_test_example(x),
                "labels": self.format_test_label(x),
            },
            remove_columns=dataset.column_names,
        )

        # Tokenize + add float labels
        dataset = dataset.map(
            lambda x: {
                **tokenizer(x["text"], add_special_tokens=False),
                "labels": x["labels"],  # preserve float labels
            },
            remove_columns=["text"],
        )
        return dataset
