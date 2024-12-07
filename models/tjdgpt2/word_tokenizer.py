from transformers import AutoTokenizer, PreTrainedTokenizer


class WordTokenizer:
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
