import torch
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel

# Not all models have the same structure
EXCEPTIONS = {
    "gpt2": lambda m: (m.transformer, m.lm_head),
}


class TJDHuggingFaceV3(PreTrainedModel):
    def __init__(self, auto_model_kwargs: dict, **kwargs):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(**auto_model_kwargs)
        self.gradient_checkpointing_enable = self.backbone.gradient_checkpointing_enable

    @property
    def config(self):
        return self.backbone.config

    def generate(
        self,
        *args,
        **kwargs,
    ):
        return self.backbone.generate(*args, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
