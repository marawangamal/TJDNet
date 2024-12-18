import os
import torch
from transformers import AutoModelForCausalLM

from models._tjd import TJD


class TJDLLAMA(TJD):
    def __init__(
        self,
        model_head: str = "base",
        vocab_size: int = 32000,
        n_embd: int = 4096,
        rank: int = 2,
        horizon: int = 8,
        positivity_func: str = "exp",
        freeze_base_model: bool = True,
        **kwargs,
    ):
        super().__init__(
            n_embd,
            vocab_size,
            rank=rank,
            horizon=horizon,
            model_head=model_head,
            positivity_func=positivity_func,
        )
        self.gradient_checkpointing_enable = self.model.gradient_checkpointing_enable

    # TODO: use attention_mask
    def get_last_hidden_state(self, input_ids, attention_mask=None):
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = transformer_outputs.last_hidden_state
        del transformer_outputs
        torch.cuda.empty_cache()
        return last_hidden_state

    def get_model(self, **model_kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            low_cpu_mem_usage=True,
        )
        transformer_model = model.model
        del model
        torch.cuda.empty_cache()

        # Set model to not require gradients
        for param in transformer_model.parameters():
            param.requires_grad = False
        return transformer_model
