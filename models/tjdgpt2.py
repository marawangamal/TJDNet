from typing import Literal

import torch
from models._tjd import TJD

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)


class TJDGPT2(TJD):
    def __init__(
        self,
        model_head: str = "base",
        vocab_size: int = 50257,
        n_embd: int = 768,
        rank: int = 2,
        horizon: int = 8,
        positivity_func: str = "exp",
        init_method: Literal["random", "pretrained"] = "random",
        freeze_base_model: bool = False,
        **kwargs,
    ):
        super().__init__(
            n_embd,
            vocab_size,
            rank=rank,
            horizon=horizon,
            model_head=model_head,
            positivity_func=positivity_func,
            model_kwargs={
                "vocab_size": vocab_size,
                "n_embd": n_embd,
            },
            freeze_base_model=freeze_base_model,
            init_method=init_method,
        )

    def freeze_base_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.transformer.h[-1].mlp.parameters():
            param.requires_grad = True

    def get_last_hidden_state(self, input_ids, attention_mask=None):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = transformer_outputs.last_hidden_state
        del transformer_outputs
        torch.cuda.empty_cache()
        return last_hidden_state

    def get_model(self, **model_kwargs):
        return GPT2LMHeadModel(GPT2Config(**model_kwargs))
