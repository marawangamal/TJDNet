from transformers import LlamaForCausalLM


class TJDLlamaModel(LlamaForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.extra_info = kwargs.get("extra_info", None)
