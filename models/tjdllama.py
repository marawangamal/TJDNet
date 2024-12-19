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
        init_method: Literal["random", "pretrained"] = "pretrained",
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
            freeze_base_model=freeze_base_model,
            init_method=init_method,
        )
        self.gradient_checkpointing_enable = self.model.gradient_checkpointing_enable

        # # Init model head
        # if init_method == "pretrained":
        # model = AutoModelForCausalLM.from_pretrained(
        #     "meta-llama/Llama-2-7b-chat-hf",
        #     low_cpu_mem_usage=True,
        # )
        # self.init_model_head_params(model.lm_head.weight)
        # del model

        # Set model to not require gradients
        # BUG: Unfreezing atleast the last layer in `self.model`` is required for proper training
        # if freeze_base_model:
        #     for param in self.model.parameters():
        #         param.requires_grad = False

    def get_pretrained_lm_head_weights(self, model):
        return model.lm_head.weight

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
