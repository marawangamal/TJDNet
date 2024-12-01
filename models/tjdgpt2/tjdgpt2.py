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
from tensorops.common import get_windowed_input_ids


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

    def _get_horizon(self, horizon: Optional[int]):
        horizon = self.horizon if horizon is None else horizon
        if horizon > self.horizon:
            raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
        return horizon

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 8,
        num_beams=1,
        do_sample=False,
        horizon: Optional[int] = None,
        **kwargs,
    ):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.
            num_beams (int, optional): Number of beams. Defaults to 1.
            do_sample (bool, optional): Whether to sample. Defaults to False.
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.

        Returns:
            torch.Tensor: Generated tokens of shape (B, `max_new_tokens`).
        """
        assert input_ids.size(0) == 1, "Only batch size 1 is supported"
        horizon = self._get_horizon(horizon)
        batch_size, _ = input_ids.size()
        n_passes = max(max_new_tokens // horizon, 1)
        output_tens = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        input_tens = input_ids

        for _ in range(n_passes):
            last_hidden_state = self._get_last_hidden_state(input_tens, **kwargs)
            sample = self.model_head.generate(
                last_hidden_state=last_hidden_state, horizon=horizon
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

        # Sequence length must be greater than horizon
        assert (
            input_ids.size(1) > self.horizon
        ), "Sequence length must be greater than horizon"

        batch_size, _ = input_ids.size()
        horizon = self.horizon if horizon is None else min(self.horizon, horizon)

        last_hidden_state = self._get_last_hidden_state(input_ids, **kwargs)
        targets = get_windowed_input_ids(input_ids, horizon=horizon).reshape(
            batch_size, -1, horizon
        )  # (B, T-H, H)

        assert targets.size(1) >= horizon, "Invalid targets"

        p_tilde, p_tilde_scale_factors = self.model_head.evaluate_at_points(
            last_hidden_state[:, :-horizon], targets
        )  # (B, T-H)

        norm_const, norm_const_scale_factors = self.model_head.get_norm_consts(
            last_hidden_state[:, :-horizon], horizon=horizon
        )  # (B, T-H)

        # Health checks
        # 1. Ensure no NaNs
        assert not torch.isnan(p_tilde).any(), "p_tilde NaN"
        assert not torch.isnan(norm_const).any(), "norm_const NaN"
        # 2. Ensure p_tilde < norm_const (if no scale factors)
        if len(p_tilde_scale_factors) == 0 and len(norm_const_scale_factors) == 0:
            assert (p_tilde < norm_const).all(), "p_tilde < norm_const"

        loss = (
            -torch.log(p_tilde + self.eps)
            + torch.log(norm_const)
            - sum([torch.log(z) for z in p_tilde_scale_factors])
            + sum([torch.log(z) for z in norm_const_scale_factors])
        ).mean()
        return loss
