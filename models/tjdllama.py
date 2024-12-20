import os
from typing import Literal
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
        init_method: Literal["random", "pretrained"] = "random",
        freeze_base_model: bool = True,
        use_memory_efficient_loss: bool = False,
        **kwargs,
    ):
        super().__init__(
            n_embd,
            vocab_size,
            rank=rank,
            horizon=horizon,
            model_head=model_head,
            positivity_func=positivity_func,
            freeze_base_model=freeze_base_model,
            init_method=init_method,
            use_memory_efficient_loss=use_memory_efficient_loss,
        )

    def get_pretrained_lm_head_weights(self):
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            low_cpu_mem_usage=True,
        )
        weights = model.lm_head.weight
        del model
        return weights

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
        return transformer_model
