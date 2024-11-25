from typing import Optional
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)

from distributions.base import BaseDist
from distributions.cp import CPDist
from distributions.full import FullDist
from distributions.mps import MPSDist
from utils.tensorops.common import get_windowed_input_ids


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
            "full": FullDist,
            "cp": CPDist,
            "mps": MPSDist,
            "base": BaseDist,
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
        max_new_tokens: int = 8,
        num_beams=1,
        do_sample=False,
        **kwargs,
    ):
        # BUG: Should only accept a batch of size 1
        batch_size, _ = input_ids.size()
        n_passes = max_new_tokens // self.horizon
        output_tens = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        input_tens = input_ids

        for _ in range(n_passes):
            last_hidden_state = self._get_last_hidden_state(input_tens, **kwargs)
            sample = self.model_head.generate(
                last_hidden_state=last_hidden_state,
            )  # (B, H)
            output_tens = torch.cat([output_tens, sample], dim=1)
            input_tens = torch.cat([input_tens, sample], dim=1)
        return output_tens

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        horizon: Optional[int] = None,
        **kwargs,
    ):
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (B, T)
            labels (torch.Tensor): Tensor of shape (B, T)
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.

        Note:
            The horizon applied in the forward pass is the minimum of the model level horizon and the horizon passed as an argument.

        Returns:
            torch.Tensor: Loss value.
        """

        horizon = self.horizon if horizon is None else min(self.horizon, horizon)

        last_hidden_state = self._get_last_hidden_state(input_ids, **kwargs)
        targets = get_windowed_input_ids(input_ids, horizon=horizon)  # (B * T-H, H)
        p_tilde, p_tilde_scale_factors = self.model_head.evaluate_at_points(
            last_hidden_state[:, :-horizon], targets, horizon=horizon
        )  # (B, T-H)

        norm_const, norm_const_scale_factors = self.model_head.get_norm_consts(
            last_hidden_state[:, :-horizon], horizon=horizon
        )  # (B, T-H)

        if len(p_tilde_scale_factors) == 0 and len(norm_const_scale_factors) == 0:
            assert (p_tilde < norm_const).all(), "p_tilde < norm_const"

        loss = (
            -torch.log(p_tilde + self.eps)
            + torch.log(norm_const)
            - sum([torch.log(z) for z in p_tilde_scale_factors])
            + sum([torch.log(z) for z in norm_const_scale_factors])
        ).mean()
        return loss
