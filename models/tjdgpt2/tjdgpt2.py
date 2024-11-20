import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)

from distributions.cp import CPDist
from utils.tensop import get_windowed_input_ids


class TJDGPT2(torch.nn.Module):
    def __init__(
        self,
        model: str = "gpt2",
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        rank: int = 2,
        eps: float = 1e-9,
        horizon: int = 8,
        positivity_func: str = "sq",
        eos_token_id: int = 50256,
        bos_token_id: int = 50256,
        pad_token_id: int = 50256,
        is_full_rank: bool = False,
    ):
        super().__init__()
        self.model_name = model
        self.model_config = {
            "model": model,
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
            "dropout": dropout,
            "rank": rank,
            "eps": eps,
            "horizon": horizon,
            "positivity_func": positivity_func,
            "eos_token_id": eos_token_id,
            "bos_token_id": bos_token_id,
            "pad_token_id": pad_token_id,
            "is_full_rank": is_full_rank,
        }
        self.horizon = horizon
        self.eps = eps
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=vocab_size,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                dropout=dropout,
                eos_token_id=eos_token_id,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
            )
        )
        self.model_head = {
            "cpgpt2": CPDist,
        }[model](
            n_embd=n_embd,
            vocab_size=vocab_size,
            rank=rank,
            horizon=horizon,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_last_hidden_state(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            **kwargs,
        )
        return transformer_outputs.last_hidden_state

    def generate(
        self,
        input_ids: torch.Tensor,
        horizon: int = 1,
        max_new_tokens: int = 8,
        num_beams=1,
        do_sample=False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.size()
        horizon = horizon if horizon is not None else self.horizon
        n_passes = max_new_tokens // horizon
        output_tens = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        input_tens = input_ids

        for _ in range(n_passes):
            last_hidden_state = self._get_last_hidden_state(input_tens, **kwargs)
            sample = self.model_head.generate(
                last_hidden_state=last_hidden_state,
                horizon=horizon,
            )  # (B, H)
            output_tens = torch.cat([output_tens, sample], dim=1)
            input_tens = torch.cat([input_tens, sample], dim=1)
        return output_tens

    def forward(self, input_ids, labels, horizon=None, **kwargs):
        last_hidden_state = self._get_last_hidden_state(input_ids, **kwargs)
        horizon = horizon if horizon is not None else self.horizon
        targets = get_windowed_input_ids(input_ids, horizon=horizon)

        assert (
            horizon == None or horizon <= self.horizon
        ), "Horizon must be <= model horizon"
        horizon = horizon if horizon is not None else self.horizon
        p_tilde, p_tilde_scale_factors = self.model_head.evaluate_at_points(
            last_hidden_state, targets
        )
        norm_const, norm_const_scale_factors = self.model_head.get_norm_const(
            last_hidden_state, horizon=horizon
        )
        loss = (
            -torch.log(p_tilde + self.eps)
            + torch.log(norm_const)
            - sum([torch.log(z) for z in p_tilde_scale_factors])
            + sum([torch.log(z) for z in norm_const_scale_factors])
        ).mean()
        return loss
