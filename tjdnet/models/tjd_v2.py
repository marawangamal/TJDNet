from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from math import e
from typing import Dict, Literal, Tuple


import torch
from wandb import config

from tjdnet.distributions import TJD_DISTS
from tjdnet.distributions._base import BaseDistConfig, BaseDistFromLinearConfig
from utils.spec_sample import speculative_sampling
from tjdnet.tensorops.common import get_windowed_input_ids_v2
from tjdnet.utils import sample_topk

from transformers import GenerationConfig


#  extend the GenerationConfig class to add the new parameters
@dataclass
class TJDGenerationConfig(GenerationConfig):
    horizon: int = 1  # Number of parallel tokens
    gen_mode: Literal["draft", "base", "speculative"] = "draft"  # Generation mode


@dataclass
class TJDConfig:
    # Model head (i.e., distribution)
    model_head: str
    model_head_config: BaseDistConfig

    # Training configuration
    init_method: Literal["random", "pretrained"] = "random"
    loss_mode: Literal["joint", "draft"] = "draft"
    joint_loss_lambda: float = 1.0  # Balance between draft and target model losses


def extend_attn(mat: torch.Tensor, horizon: int = 1):
    """Extend attention mask to include the horizon."""
    return torch.cat(
        (
            mat,
            torch.ones(
                (mat.size(0), horizon),
                device=mat.device,
            ),
        ),
        dim=1,
    )


class TJD(ABC, torch.nn.Module):
    """Joint Distribution Transformer model."""

    def __init__(self, config: TJDConfig, **kwargs):
        self.tjd_config = config
        self.backbone, self.lm_head = self.get_model()

        if config.init_method == "pretrained":
            self.mhead = TJD_DISTS[config.model_head].from_linear(
                linear=self.lm_head,
                config=BaseDistFromLinearConfig(
                    horizon=config.model_head_config.horizon,
                    rank=config.model_head_config.rank,
                    param_net=config.model_head_config.param_net,
                ),
            )
        else:
            self.mhead = TJD_DISTS[config.model_head](config=config.model_head_config)

        # Trainer compatibility
        self.gradient_checkpointing_enable = self.backbone.gradient_checkpointing_enable

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def get_model(self) -> Tuple[torch.nn.Module, torch.nn.Linear]:
        pass

    def generate(
        self,
        inputs: torch.Tensor,
        generation_config: TJDGenerationConfig,
        **kwargs,  # will be passed to the forward method
    ):
        """Generate sequences given an input tensor.

        Args:
            x (torch.Tensor): Previous tokens of shape (B, T)
            generation_config (TJDGenerationConfig): Generation configuration.

        Returns:
            torch.Tensor: Generated tokens of shape (B, T_out). T_out <= T + max_new_tokens if stop_token is used. Otherwise, T_out = T + max_new_tokens.
        """

        # ==== Input validation
        input_validation_checks = [
            {
                "test": lambda: inputs > 0,
                "msg": "Input tokens must be positive",
            }
        ]
        for check in input_validation_checks:
            assert check["test"](), check["msg"]
        # ====

        B, H, T = self.horizon, inputs.size(0), inputs.size(1)
        device = inputs.device
        temp_token = -100  # Temporary token for padding

        gen_kwargs = {
            "top_k": generation_config.top_k if generation_config.do_sample else 1,
        }

        # Initialize output with input_ids
        y_out = torch.full(
            (B, inputs.size(1) + generation_config.max_new_tokens),
            fill_value=temp_token,
            dtype=torch.long,
            device=device,
        )  # (B, T + N)
        y_out[:, : inputs.size(1)] = inputs

        accept_rate_metrics = {
            "tokens_proposed": 0,
            "tokens_accepted": 0,
        }

        with torch.no_grad():
            t = 0
            while t < generation_config.max_new_tokens:
                H = min(
                    H, generation_config.max_new_tokens - t
                )  # Handle case when < horizon tokens are left

                mask_active = torch.ones(B, device=device).bool()
                if generation_config.eos_token_id is not None:
                    mask_active = ~torch.any(
                        y_out[:, inputs.size(1) :] == generation_config.eos_token_id,
                        dim=1,
                    )

                # Exit if all sequences are done
                if not mask_active.any():
                    break

                # Get hidden state
                y_prime = y_out[mask_active, : T + t]  # (B', T + t)
                h_last = self.backbone(input_ids=y_prime, **kwargs)[:, -1]

                # Sample
                if generation_config.gen_mode == "speculative":
                    y_cand, qy = self.mhead.sample(
                        h_last,
                        horizon=H,
                        do_sample=generation_config.do_sample,
                        top_k=generation_config.top_k,
                    )
                    attn_mask = kwargs.get("attention_mask", None)
                    if attn_mask is not None:
                        attn_mask = extend_attn(
                            attn_mask[mask_active, : T + t], horizon=H
                        )
                    else:
                        raise Warning(
                            "Attention mask not provided. Generation may not work as expected."
                        )
                    py = self.backbone(
                        input_ids=torch.cat((inputs, y_cand), dim=1),  # (B', T + t + H)
                        attention_mask=attn_mask,
                        **kwargs,
                    )
                    y_hat, n_matches = speculative_sampling(
                        candidate_input_ids=y_cand,
                        candidate_logits=qy,
                        candidate_length=H,
                        new_logits=py,
                        is_done_candidate=False,
                    )  # (B', H') -- H' <= H_tgt if not all tokens are accepted

                    accept_rate_metrics["tokens_proposed"] += H
                    accept_rate_metrics["tokens_accepted"] += y_hat.size(1) - 1

                elif generation_config.gen_mode == "draft":
                    y_hat, _ = self.mhead.sample(
                        h_last,
                        horizon=H,
                        do_sample=generation_config.do_sample,
                        top_k=generation_config.top_k,
                    )  # (B', H_tgt)

                elif generation_config.gen_mode == "base":
                    logits_y = self.lm_head(h_last)
                    y_hat = sample_topk(logits_y, **gen_kwargs).squeeze(-1)

                else:
                    raise ValueError(
                        f"Invalid generation mode: {generation_config.gen_mode}"
                    )

                # Append new tokens
                H_sampled = y_hat.size(1)  # Number of tokens sampled
                time_step_abs = inputs.size(1) + t
                y_out[
                    mask_active,
                    time_step_abs : time_step_abs + H_sampled,
                ] = y_hat

                t += H_sampled

        # Replace stop tokens with padding
        if generation_config.eos_token_id:
            y_out[y_out == temp_token] = generation_config.eos_token_id
            stop_mask = (y_out == generation_config.eos_token_id).float()  # (B, T_out)
            y_out[torch.cumsum(stop_mask, dim=1) >= 1] = generation_config.eos_token_id

        return y_out, accept_rate_metrics

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,  # NOTE: needed for compatibility with Trainer
        attention_mask=None,
        reduce="mean",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (B, T)
            labels (torch.Tensor): Tensor of shape (B, T)
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.
            reduce (str, optional): Reduction method. Defaults to "mean".
            use_memory_efficient_loss (bool, optional): Whether to use memory efficient loss computation. Defaults to False.

        Note:
            horizon must be less than or equal to the model horizon specified during initialization.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - loss (torch.Tensor): Reduced loss value of shape (B,)
                - nll (torch.Tensor): Reduced negative log likelihood of shape (B,)
                - loss_scale (torch.Tensor): Loss scaling factor, scalar tensor of value 1/rank

        """

        # ==== Input validation
        input_validation_checks = [
            {
                "test": lambda: input_ids.size(1) > self.horizon,
                "msg": "Sequence length must be greater than horizon",
            }
        ]
        for check in input_validation_checks:
            assert check["test"](), check["msg"]
        # ====

        reduce_fn = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x,
        }[reduce]

        B, _ = input_ids.size()
        H = self.horizon

        h = self.backbone(input_ids, attention_mask=attention_mask)

        # 1. Create targets
        y_true = get_windowed_input_ids_v2(input_ids, horizon=self.horizon).reshape(
            B, -1, H
        )  # (B, T-H, H)

        # 2. Downsample if `use_memory_efficient_loss` is True
        # This is a hack to avoid memory issues. The shift is random to align the expectation of the loss
        # with the non-downsampled version.
        h_ds = h[:, :-H]  # (B, T-H, D)
        y_true_ds = y_true  # (B, T-H, H)
        if self.use_memory_efficient_loss:
            shift = torch.randint(0, H, (1,)).item()
            h_ds = h_ds[:, shift::H]  # (B, T-H // H, D)
            y_true_ds = y_true[:, shift::H]  # (B, T-H // H, H)

        # 3a. Compute mhead loss (draft model)
        # (B, T') i.e., maybe downsampled
        loss_mhead = self.mhead.compute_loss(
            x=h_ds.reshape(-1, self.n_embd),
            y=y_true_ds.reshape(-1, H),
        ).reshape(B, -1)
        loss_tot = loss_mhead.sum(-1)

        # 3b. Compute lm_head loss (target model)
        if self.loss_mode == "joint":
            log_probs_lm_head = self.tgt_model_head(h[:, :-1])  # (B, T-1, V)
            # (B, T) -> (B, T-1)
            y_true = get_windowed_input_ids_v2(
                input_ids,
                horizon=1,
            ).squeeze(-1)
            loss_lm_head = (
                torch.nn.functional.cross_entropy(
                    log_probs_lm_head.view(-1, self.vocab_size),
                    y_true.view(-1),
                    reduction="none",
                )
                .reshape(B, -1)  # (B, T-1)
                .sum(-1)
            )
            loss_tot = loss_tot + self.joint_loss_lambda * loss_lm_head  # (B,)

        # NLL must be computed on downsampled seq.
        nll = loss_mhead if self.use_memory_efficient_loss else loss_mhead[:, ::H]
        return {
            "loss": reduce_fn(loss_tot),
            "nll": reduce_fn(nll.sum(dim=-1)),
            "loss_scale": torch.tensor(1 / self.rank).to(loss_mhead.device),
        }
