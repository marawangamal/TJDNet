from abc import ABC, abstractclassmethod, abstractmethod
from typing import Union
from git import Optional
import torch

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class BaseChatTemplate(ABC):
    @classmethod
    def check_answer(cls, answer_pred_unp: str, answer_true_unp: str, eos_token: str):
        try:
            answer_pred = cls.safe_parse(answer_pred_unp, eos_token)
            answer_true = cls.safe_parse(answer_true_unp, eos_token)
            if not (answer_pred and answer_true):
                return False
            return answer_pred == answer_true
        except Exception:
            return False

    @classmethod
    @abstractmethod
    def get_sample_prompt(cls, is_few_shot: bool = False) -> str:
        """Returns a sample prompt."""
        pass

    @classmethod
    @abstractmethod
    def safe_parse(cls, generation: str, eos_token: str) -> Optional[float]:
        pass

    @classmethod
    @abstractmethod
    def parse_answer(cls, generation: str, eos_token: str) -> float:
        pass


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
