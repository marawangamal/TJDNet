from dataclasses import dataclass
from abc import ABC, abstractmethod
from math import e
from typing import Dict, Literal, Optional, Tuple


import torch
from wandb import config

from tjdnet.distributions import TJD_DISTS
from tjdnet.distributions._tjdist import BaseDistConfig, BaseDistFromLinearConfig
from tjdnet.spec_sample import spec_sample
from tjdnet.tensorops.common import get_windowed_input_ids_v2
from tjdnet.utils import sample_topk


#  extend the GenerationConfig class to add the new parameters
@dataclass
class TJDGenerationConfig:
    gen_mode: Literal["draft", "base", "speculative"] = "draft"  # Generation mode
    do_sample: bool = False
    top_k: int = 1  # Top-k sampling
    max_new_tokens: int = 32  # Maximum number of new tokens to generate
    eos_token_id: Optional[int] = None  # End of sequence token ID
    horizon: Optional[int] = None  # Horizon for the model head


@dataclass
class TJDConfig:
    # Model head (i.e., distribution)
    model_head: str
    model_head_config: BaseDistConfig

    # Training configuration
    init_method: Literal["random", "pretrained"] = "random"
    loss_mode: Literal["joint", "draft"] = "draft"
    joint_loss_lambda: float = 1.0  # Balance between draft and target model losses
    use_memory_efficient_loss: bool = True  # Use memory-efficient loss computation


def extend_attn_mask(mask: torch.Tensor, horizon: int = 1):
    """Extend attention mask to include the horizon."""
    return torch.cat(
        (
            mask,
            torch.ones(
                (mask.size(0), horizon),
                device=mask.device,
            ),
        ),
        dim=1,
    )


class TJD(ABC, torch.nn.Module):
    """Base class for TJD models.

    Args:
        config (TJDConfig): Configuration object for the model.
        **kwargs: Additional keyword arguments.

    Note:
        This class is an abstract base class and should not be instantiated directly.
        Subclasses must implement the `get_model` method to provide the backbone model and lm_head.
    """

    def __init__(self, config: TJDConfig, **kwargs):
        super().__init__()
        self.tjd_config = config
        self.backbone, self.lm_head = self.get_model()

        # Set dims
        self.horizon = config.model_head_config.horizon
        self.n_embd = config.model_head_config.param_net.in_dim
        self.vocab_size = config.model_head_config.vocab_size

        # Initialize model head
        if config.init_method == "pretrained":
            self.mhead = TJD_DISTS[config.model_head].from_pretrained(
                linear=self.lm_head,
                config=BaseDistFromLinearConfig(
                    horizon=config.model_head_config.horizon,
                    rank=config.model_head_config.rank,
                    param_net=config.model_head_config.param_net,
                ),
            )
        else:
            self.mhead = TJD_DISTS[config.model_head](config=config.model_head_config)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _run_checks(input_validation_checks: list):
        for check in input_validation_checks:
            assert check["test"](), check["msg"]

    @abstractmethod
    def get_model(self) -> Tuple[torch.nn.Module, torch.nn.Linear]:
        """Get the model and the linear layer.

        Returns:
            tuple: Tuple containing
                - backbone (torch.nn.Module): Backbone model.
                - lm_head (torch.nn.Linear): Linear layer for the model head.
        """
        pass

    @abstractmethod
    def forward_backbone(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the backbone model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing
                - h_targ (torch.Tensor): Hidden state for use by target head
                - h_draft (torch.Tensor): Hidden state for use by draft head

        """
        pass

    def prob_y_bar_x_backbone(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Compute probabilities P(yt|x, y1:t-1) for all t timesteps.

        Args:
            input_ids (torch.Tensor): Input ids. Shape (B, T).
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Shape (B, T). Defaults to None.

        Returns:
            torch.Tensor: Probabilities of shape (B, T, V).
        """
        h_targ, _ = self.forward_backbone(
            input_ids=x,
            attention_mask=attn_mask,
            **kwargs,
        )
        logits = self.lm_head(h_targ)
        if return_logits:
            return logits
        return torch.nn.functional.softmax(logits, dim=-1)

    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: TJDGenerationConfig,
        return_full_text: bool = False,
        **kwargs,
    ):
        """Generate sequences given an input tensor.

        Args:
            inputs (torch.Tensor): Previous tokens of shape (B, T)
            generation_config (TJDGenerationConfig): Generation configuration.

        Returns:
            torch.Tensor: Generated tokens of shape (B, T_out). T_out <= T + max_new_tokens if stop_token is used. Otherwise, T_out = T + max_new_tokens.
        """

        # ==== Input validation
        input_validation_checks = [
            {
                "test": lambda: torch.all(input_ids > 0),
                "msg": "Input tokens must be positive",
            }
        ]
        self._run_checks(input_validation_checks)
        # ====

        B, T, H = (
            input_ids.size(0),
            input_ids.size(1),
            generation_config.horizon or self.horizon,
        )
        device = input_ids.device
        temp_token = -100  # Temporary token for padding

        gen_kwargs = {
            "top_k": generation_config.top_k if generation_config.do_sample else 1,
        }
        sample_fn = lambda x: sample_topk(x, **gen_kwargs).squeeze(-1)

        # Initialize output with input_ids
        y_out = torch.full(
            (B, input_ids.size(1) + generation_config.max_new_tokens),
            fill_value=temp_token,
            dtype=torch.long,
            device=device,
        )  # (B, T + N)
        y_out[:, : input_ids.size(1)] = input_ids

        accept_rate_metrics = {
            "tokens_generated": 0,
            "tokens_accepted": 0,
        }

        with torch.no_grad():
            t = 0
            while t < generation_config.max_new_tokens:
                H = min(
                    H, generation_config.max_new_tokens - t
                )  # Handle case when < horizon tokens are left

                if t >= 70:
                    dummy = 1
                    pass

                mask_active = torch.ones(B, device=device).bool()
                if generation_config.eos_token_id is not None:
                    # mask_active = ~torch.any(
                    #     y_out[:, inputs.size(1) :] == generation_config.eos_token_id,
                    #     dim=1,
                    # )
                    mask_active = ~torch.any(
                        y_out == torch.tensor(generation_config.eos_token_id), dim=1
                    )

                # Exit if all sequences are done
                if not mask_active.any():
                    break

                # Get hidden state
                x = y_out[mask_active, : T + t]  # (B', T + t)
                _, h_draft = self.forward_backbone(input_ids=x)
                h_last_draft = h_draft[:, -1]

                # Sample
                if generation_config.gen_mode == "speculative":

                    def model_p(y):
                        attn_mask = kwargs.get("attention_mask", None)
                        if attn_mask is not None:
                            attn_mask = extend_attn_mask(attn_mask, horizon=H)
                        return self.prob_y_bar_x_backbone(
                            x=torch.cat((x, y), dim=1),
                            attn_mask=attn_mask,
                        )[:, x.size(1) - 1 : -1]

                    def model_q():
                        return self.mhead.sample(
                            h_last_draft,
                            horizon=H,
                            sample_fn=sample_fn,
                        )

                    y_hat, n_accept = spec_sample(
                        model_p=model_p,  # (B', H, V)
                        model_q=model_q,  # None -> (B', H), (B', H, V)
                        sample_fn=sample_fn,
                    )

                    accept_rate_metrics["tokens_generated"] += H
                    accept_rate_metrics["tokens_accepted"] += n_accept

                elif generation_config.gen_mode == "draft":
                    # y_hat ~ p(y | x, y1:t-1)
                    y_hat, _ = self.mhead.sample(
                        h_last_draft,
                        horizon=H,
                        sample_fn=sample_fn,
                    )  # (B', H_tgt)

                elif generation_config.gen_mode == "base":
                    logits_y = self.lm_head(h_last_draft)
                    y_hat = sample_fn(logits_y).unsqueeze(-1)  # (B', 1)

                else:
                    raise ValueError(
                        f"Invalid generation mode: {generation_config.gen_mode}"
                    )

                # Append new tokens
                H_sampled = y_hat.size(1)  # Number of tokens sampled
                y_out[mask_active, x.size(1) : x.size(1) + H_sampled] = y_hat
                t += H_sampled

        # Replace stop tokens with padding
        if generation_config.eos_token_id:
            y_out[y_out == temp_token] = generation_config.eos_token_id
            stop_mask = (y_out == generation_config.eos_token_id).float()  # (B, T_out)
            y_out[torch.cumsum(stop_mask, dim=1) >= 1] = generation_config.eos_token_id

        y_out = y_out if return_full_text else y_out[:, input_ids.size(1) :]
        return y_out, accept_rate_metrics

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
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
                "test": lambda: input_ids.size(1) >= self.horizon
                and input_ids.size(1) % self.horizon == 0,
                "msg": f"Input sequence length must be greater than or equal to {self.horizon} and divisible by {self.horizon}",
            }
        ]
        self._run_checks(input_validation_checks)
        # ====

        reduce_fn = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x,
        }[reduce]

        B, _ = input_ids.size()
        H = self.horizon

        h_targ, h_draft = self.forward_backbone(
            input_ids, attention_mask=attention_mask
        )  # (B, T, D), (B, T, D)

        # 1. Create targets
        y_true = get_windowed_input_ids_v2(input_ids, horizon=self.horizon).reshape(
            B, -1, H
        )  # (B, T-H, H)

        # 2. Downsample if `use_memory_efficient_loss` is True
        # This is a hack to avoid memory issues. The shift is random to align the expectation of the loss
        # with the non-downsampled version.
        h_draft_ds = h_draft[:, :-H]  # (B, T-H, D)
        y_true_ds = y_true  # (B, T-H, H)
        if self.tjd_config.use_memory_efficient_loss:
            shift = torch.randint(0, H, (1,)).item()
            h_draft_ds = h_draft_ds[:, shift::H]  # (B, T-H // H, D)
            y_true_ds = y_true[:, shift::H]  # (B, T-H // H, H)

        # 3a. Compute mhead loss (draft model)
        # (B, T') i.e., maybe downsampled
        loss_mhead = self.mhead(
            x=h_draft_ds.reshape(-1, self.n_embd),
            y=y_true_ds.reshape(-1, H),
        ).reshape(
            B, -1
        )  # (B, T-H // H)
        loss_tot = loss_mhead.sum(-1)  # (B,)

        # 3b. Compute lm_head loss (target model)
        loss_target = torch.zeros(1, device=input_ids.device)
        loss_draft = loss_tot
        if self.tjd_config.loss_mode == "joint":
            log_probs_lm_head = self.lm_head(h_targ[:, :-1])  # (B, T-1, V)
            # (B, T) -> (B, T-1)
            y_true = get_windowed_input_ids_v2(
                input_ids,
                horizon=1,
            ).squeeze(-1)
            loss_lm_head = (
                torch.nn.functional.cross_entropy(
                    log_probs_lm_head.reshape(-1, self.vocab_size),
                    y_true.reshape(-1),
                    reduction="none",
                )
                .reshape(B, -1)  # (B, T-1)
                .sum(-1)  # (B,)
            )
            loss_target = loss_lm_head
            loss_tot = (
                loss_tot + self.tjd_config.joint_loss_lambda * loss_target
            )  # (B,)

        # NLL must be computed on downsampled seq.
        nll = (
            loss_mhead
            if self.tjd_config.use_memory_efficient_loss
            else loss_mhead[:, ::H]
        ).sum(dim=-1)
        return {
            "loss": reduce_fn(loss_tot),
            "nll": reduce_fn(nll),
            "loss_draft": reduce_fn(loss_draft),
            "loss_target": reduce_fn(loss_target),
        }
