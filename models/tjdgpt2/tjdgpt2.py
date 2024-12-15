from models.tjdgpt2.tjd import TJD

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
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        rank: int = 2,
        horizon: int = 8,
        positivity_func: str = "exp",
        eos_token_id: int = 50256,
        bos_token_id: int = 50256,
        pad_token_id: int = 50256,
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
                "n_layer": n_layer,
                "n_head": n_head,
                "dropout": dropout,
                "eos_token_id": eos_token_id,
                "bos_token_id": bos_token_id,
                "pad_token_id": pad_token_id,
            },
        )

    # TODO: use attention_mask
    def get_last_hidden_state(self, input_ids, attention_mask=None):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
        )
        return transformer_outputs.last_hidden_state

    def get_model(self, **model_kwargs):
        return GPT2LMHeadModel(GPT2Config(**model_kwargs))
