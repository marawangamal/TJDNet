# sharegptv2.py


import os
from transformers import AutoTokenizer
from datasets import load_dataset

from dataloaders._base import HF_CACHE_DIR, setup
from dataloaders.common import BaseChatTemplate, group_texts


class ChatTemplateShareGPT(BaseChatTemplate):
    TEMPLATE = """[QUESTION]{question}\n[ANSWER]{answer}"""
    TEMPLATE_ASSISTANT_TURN = """[ANSWER]{message}"""
    TEMPLATE_HUMAN_TURN = """[QUESTION]{message}"""
    TEMPLATE_FEW_SHOT = """
    You are a helpful assistant that answers questions step by step.

    FORMAT INSTRUCTIONS:
    1. Show all your work with clear explanations
    2. For each calculation, use the format: <<calculation=result>>result
    3. End every answer with: #### [numerical_answer_only]

    EXAMPLE:
    [QUESTION] In Java, I want to replace string like 'This is a new {{ object }} at {{ place }}' with a Map, {{object: "student", "point 3, 4"}}, and get a result "This is a new student at point 3, 4". How can I do?
    [ANSWER]  You can use the `String.format()` method in Java to replace placeholders in a string with values from a map. Here's an example code snippet that demonstrates how you can achieve this:
    ```java
    import java.util.HashMap;
    import java.util.Map;

    public class StringReplaceExample {{
        public static void main(String[] args) {{
            String input = "This is a new {{object}} at {{place}}";
            Map<String, String> replacements = new HashMap<>();
            replacements.put("object", "student");
            replacements.put("place", "point 3, 4");

            String output = replacePlaceholders(input, replacements);
            System.out.println(output);
        }}

        public static String replacePlaceholders(String input, Map<String, String> replacements) {{
            for (Map.Entry<String, String> entry : replacements.entrySet()) {{
                String placeholder = "{{" + entry.getKey() + "}}";
                String replacement = entry.getValue();
                input = input.replace(placeholder, replacement);
            }}
            return input;
        }}
    }}
    ```
    In this example, we define the input string as "This is a new {{object}} at {{place}}", and create a `HashMap` called `replacements` that maps the placeholders "object" and "place" to their corresponding values "student" and "point 3, 4". We then pass these values to the `replacePlaceholders()` method, which iterates over the entries in the `replacements` map and replaces each placeholder in the input string with its corresponding value using the `replace()` method. Finally, the `replacePlaceholders()` method returns the modified string, which is printed to the console.

    The output of this program will be:
    ```csharp
    This is a new student at point 3, 4
    ```
    Note that you can modify the `replacements` map to include additional placeholders and their corresponding values, and the `replacePlaceholders()` method will automatically replace them in the input string.

    Now solve the following problem using the exact format shown above:
    [QUESTION] {question}
    [ANSWER] 
    """

    @classmethod
    def get_sample_prompt(cls, is_few_shot: bool = False):
        tmp = cls.TEMPLATE if not is_few_shot else cls.TEMPLATE_FEW_SHOT
        return tmp.format(
            question='complete the following code from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """',
            answer="",
        )

    @classmethod
    def safe_parse(cls, generation: str, eos_token: str):
        raise NotImplementedError("safe_parse is not implemented for ShareGPT. ")


def parse_conversation(example, eos_token="<|endoftext|>"):
    text = ""
    for msg in example["conversations"]:
        assert msg["from"] in ["gpt", "human"], "Invalid message sender"
        is_assistant = msg["from"] == "gpt"
        tmpl = (
            ChatTemplateShareGPT.TEMPLATE_ASSISTANT_TURN
            if is_assistant
            else ChatTemplateShareGPT.TEMPLATE_HUMAN_TURN
        )
        text += tmpl.format(message=msg["value"])
        if is_assistant:
            text += eos_token
    return {"text": text}


def load_sharegpt(
    tokenizer,
    input_seq_len,
    test_size=0.01,
    max_num_samples=68000,
    **kwargs,
):

    dataset = (
        load_dataset(
            "Aeala/ShareGPT_Vicuna_unfiltered",
            split="train",
            cache_dir=os.path.join(HF_CACHE_DIR, "datasets"),
        ),
    )
    dataset = dataset.select(range(max_num_samples))

    # Process the selected samples
    dataset = dataset.map(
        lambda x: parse_conversation(x, tokenizer.eos_token),
        remove_columns=dataset.column_names,
    )

    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text"],
    )
    dataset = dataset.map(lambda x: group_texts(x, input_seq_len), batched=True)
    dataset = dataset.train_test_split(test_size=test_size)
    return {
        "train": dataset["train"],
        "eval": dataset["test"],
    }


# Usage example:
if __name__ == "__main__":
    setup()

    # Example usage
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dataset = load_sharegpt(tokenizer=tokenizer, input_seq_len=512, max_num_samples=100)

    print(f"\nDataset sizes:")
    print(f"Train: {len(dataset['train'])} sequences")
    print(f"Eval: {len(dataset['eval'])} sequences")
    print(f"EOS token: {tokenizer.eos_token}")

    # Find an example with EOS token
    for batch_idx, example in enumerate(dataset["train"]):
        if tokenizer.eos_token in example["input_ids"]:
            break
    print(f"\n{batch_idx}nd batch:")
    decoded_text = tokenizer.decode(example["input_ids"])
    print(decoded_text)
