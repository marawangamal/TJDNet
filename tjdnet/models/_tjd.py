from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Tuple

from peft import LoraConfig, TaskType, get_peft_model  # type: ignore


import torch

from tjdnet.distributions import TJD_DISTS
from tjdnet.distributions._base import BaseDistConfig, BaseDistFromLinearConfig
from tjdnet.tensorops.common import get_windowed_input_ids
from tjdnet.utils import mem_check, sample_topk, spec_sample_v2


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


@dataclass
class TJDConfig:
    """Configuration for Joint Distribution Transformer model.

    Attributes:
        base_dist (BaseDistConfig): Base distribution configuration containing core parameters
            like vocabulary size, embedding dimension, and distribution parameters.

        # Model Architecture
        model_head (str): Language model head type. Options:
            - "base": No joint distribution (default)
            - "cp": CP tensor decomposition
            - "mps": MPS tensor decomposition
        auto_model_kwargs (Dict): Additional arguments passed to the base model.

        # Training Configuration
        init_method (Literal["random", "pretrained"]): Model initialization method.
            - "random": Initialize weights randomly (default)
            - "pretrained": Load pretrained weights
        freeze_base_model (bool): If True, freezes the base model parameters.
        use_memory_efficient_loss (bool): If True, uses memory-efficient loss computation.
        use_speculative_sampling (bool): If True, uses speculative sampling during generation.

        # Numerical Parameters
        eps (float): Small value for numerical stability in computations.
    """

    # Core configuration
    base_dist: BaseDistConfig

    # Model architecture
    model_head: str = "base"
    auto_model_kwargs: Dict = field(default_factory=dict)

    # Training configuration
    init_method: Literal["random", "pretrained"] = "random"
    train_mode: Literal["full", "last", "lora"] = "full"
    loss_mode: Literal["joint", "draft"] = "draft"
    lora_rank: int = 512
    use_memory_efficient_loss: bool = False
    use_speculative_sampling: bool = False
    use_attn_layer: bool = False

    # Numerical parameters
    eps: float = 1e-9
    joint_loss_lambda: float = 0.1

    # Generation parameters
    eos_token_id: Optional[int] = None


class TJD(ABC, torch.nn.Module):
    """Joint Distribution Transformer model."""

    def __init__(self, config: TJDConfig, **kwargs):
        """Initialize the TJD model.

        Args:
            config (TJDConfig): Complete model configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Add all under self.config
        # self.config = config

        # Initialize core parameters
        # TODO: want to put these all under an .config but `config` is already used by Trainer
        self.rank = config.base_dist.rank
        self.horizon = config.base_dist.horizon
        self.vocab_size = config.base_dist.vocab_size
        self.train_mode = config.train_mode
        self.loss_mode = config.loss_mode
        self.n_embd = config.base_dist.param_net.in_dim
        self.joint_loss_lambda = config.joint_loss_lambda
        self.eps = config.eps
        self.use_attn_layer = config.use_attn_layer
        self.use_memory_efficient_loss = config.use_memory_efficient_loss
        self.use_speculative_sampling = config.use_speculative_sampling

        # DEBUG: LoraConfig
        if config.train_mode == "full":
            self.backbone, self.tgt_model_head = self.get_model(
                **config.auto_model_kwargs
            )
        elif config.train_mode == "last":
            self.backbone, self.tgt_model_head = self.get_model(
                **config.auto_model_kwargs
            )
            self.freeze_base_model()
        elif config.train_mode == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            backbone, self.tgt_model_head = self.get_model(**config.auto_model_kwargs)
            self.backbone = get_peft_model(backbone, peft_config)  # type: ignore
        else:
            raise ValueError(f"Invalid train_mode: {config.train_mode}")

        self.tjd_attn = (
            torch.nn.MultiheadAttention(
                embed_dim=self.n_embd,
                num_heads=1,
                batch_first=True,
            )
            if self.use_attn_layer
            else None
        )

        # Handle model initialization
        if config.init_method == "pretrained":
            self.model_head = TJD_DISTS[config.model_head].from_linear(
                self.tgt_model_head,
                BaseDistFromLinearConfig(
                    horizon=config.base_dist.horizon,
                    rank=config.base_dist.rank,
                    param_net=config.base_dist.param_net,
                ),
            )
        else:
            self.model_head = TJD_DISTS[config.model_head](config.base_dist)

        # Trainer compatibility
        self.gradient_checkpointing_enable = self.backbone.gradient_checkpointing_enable

    @property
    def param_dict(self):
        n_total_params = sum(p.numel() for p in self.parameters())
        n_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        # Get human readable format (in millions)
        n_trainable_params = f"{n_trainable_params / 1e6:.2f}M"
        n_total_params = f"{n_total_params / 1e6:.2f}M"

        return {
            "Trainable Params (M)": n_trainable_params,
            "Total Params (M)": n_total_params,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    # TODO: rename to get_runtime_horizon or is_valid_horizon
    def _get_horizon(self, horizon: Optional[int]) -> int:
        """Get the horizon value. Prevents runtime horizon exceeding the model horizon.

        Args:
            horizon (Optional[int]): Candidate horizon value.

        Raises:
            ValueError: If the horizon is greater than the model horizon.

        Returns:
            int: Horizon value.
        """
        horizon = self.horizon if horizon is None else horizon
        if horizon > self.horizon:
            raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
        return horizon

    @abstractmethod
    def get_model(self, **kwargs) -> Tuple[torch.nn.Module, torch.nn.Linear]:
        """Get the torch model to be modified.

        Returns:
            tuple[torch.nn.Module, torch.nn.Linear]:
                - Pretrained model backbone.
                - Pretrained model head. (i.e., linear layer for unembedding)
        """
        pass

    @abstractmethod
    def get_last_hidden_state(
        self, input_ids: torch.Tensor, attention_mask=None
    ) -> torch.Tensor:
        """Get the last hidden state of the model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (B, T).
            attention_mask ([type], optional): Attention mask of shape (B, T). Defaults to None.

        Returns:
            torch.Tensor: Last hidden state of shape (B, T, n_embd).
        """
        pass

    def get_attn_last_hidden_state(
        self, input_ids: torch.Tensor, attention_mask=None
    ) -> torch.Tensor:
        """Get the last hidden state of the model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (B, T).
            attention_mask ([type], optional): Attention mask of shape (B, T). Defaults to None.

        Returns:
            torch.Tensor: Last hidden state of shape (B, T, n_embd).
        """
        hidden_states = self.get_last_hidden_state(input_ids, attention_mask)
        if self.use_attn_layer and self.tjd_attn is not None:
            attn_mask = None
            if attention_mask is not None:
                attn_mask = (
                    ((1 - attention_mask).bool())
                    .unsqueeze(1)
                    .expand(-1, input_ids.size(1), -1)
                )
                causal_mask = torch.triu(
                    torch.ones(
                        input_ids.size(1), input_ids.size(1), device=input_ids.device
                    ),
                    diagonal=1,
                )  # (T, T)
                attn_mask = (
                    attn_mask
                    | causal_mask.unsqueeze(0).expand(input_ids.size(0), -1, -1).bool()
                )  # (B, T, T)
            attn_output, _ = self.tjd_attn(
                hidden_states,
                hidden_states,
                hidden_states,
                is_causal=True,
                attn_mask=attn_mask,
            )
            return attn_output  # (B, T, n_embd)
        return hidden_states

    def freeze_base_model(self):
        """Freeze the base model."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    # TODO: use x instead of active_seqs
    def generate(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        stop_token: Optional[int] = None,
        max_new_tokens: int = 8,
        do_sample: bool = True,
        horizon: Optional[int] = None,
        top_k: int = 50,
        return_new_tokens: bool = True,
        **kwargs,
    ):
        """Generate sequences given an input tensor.

        Args:
            x (torch.Tensor): Previous tokens of shape (B, T)
            attention_mask (Optional[torch.Tensor], optional): Attention mask of shape (B, T). Defaults to None.
            stop_token (Optional[int], optional): Stop token for generation. Defaults to None.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.
            do_sample (bool, optional): Whether to sample. Defaults to False.
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.
            top_k (int, optional): Top k sampling. Defaults to 50.
            return_new_tokens (bool, optional): Return only new tokens by default. Defaults to True.

        Returns:
            torch.Tensor: Generated tokens of shape (B, T_out). T_out <= T + max_new_tokens if stop_token is used. Otherwise, T_out = T + max_new_tokens.
        """

        assert torch.all(x >= 0), "Input tokens must be positive"
        assert stop_token is None or stop_token > 0, "Stop token must be positive"
        if attn_mask is not None:
            assert torch.all(
                torch.tensor(x.shape == attn_mask.shape)
            ), "Shape mismatch between input_ids and attention_mask"

        horizon = self._get_horizon(horizon)
        batch_size = x.size(0)
        device = x.device

        temp_token = -100  # Temporary token for padding

        # Initialize output with input_ids
        y_out = torch.full(
            (batch_size, x.size(1) + max_new_tokens),
            fill_value=temp_token,
            dtype=torch.long,
            device=device,
        )  # (B, T + N)
        y_out[:, : x.size(1)] = x

        accept_rate_metrics = {
            "tokens_proposed": 0,
            "tokens_accepted": 0,
        }

        with torch.no_grad():
            time_step = 0
            while time_step < max_new_tokens:
                time_step_prime = x.size(1) + time_step
                # Mask out completed seqs (i.e., seqs that encountered stop_token)
                mask_active = (
                    ~torch.any(y_out[:, x.size(1) :] == stop_token, dim=1)
                    if stop_token is not None
                    else torch.ones(batch_size, device=device).bool()
                )  # (B,)
                # Exit if all sequences are done
                if not mask_active.any():
                    break

                # Get currently active sequences
                y_prime = y_out[mask_active, :time_step_prime]  # (B', T + time_step)
                attn_mask_prime = (
                    extend_attn(attn_mask[mask_active], time_step)
                    if attn_mask is not None
                    else None
                )  # (B', T + time_step)

                # Generate next tokens
                h = self.get_attn_last_hidden_state(
                    input_ids=y_prime, attention_mask=attn_mask_prime
                )  # (B', T + time_step, D)

                horizon_target = min(
                    horizon, max_new_tokens - time_step
                )  # Handle case when < horizon tokens are left

                def model_p(y: torch.Tensor):
                    """Target model p(y|x)."""
                    # Signature: model_p: y -> p(y|x)
                    # Shape of y: (B', H')
                    # Shape of p(y|x): (B', H', V)
                    x = y_prime  # (B', T + time_step)
                    h = self.get_attn_last_hidden_state(
                        input_ids=torch.cat((x, y), dim=1),
                        attention_mask=attn_mask_prime,
                    )  # (B', T + time_step + H', D)
                    py_bar_x_prime = self.tgt_model_head(h)  # (B', t' + H', V)
                    return torch.softmax(py_bar_x_prime[:, time_step_prime:], dim=-1)

                if self.use_speculative_sampling:
                    y_hat = spec_sample_v2(
                        # model_p: y -> p(y|x). Shape: (B', H') -> (B', H', V)
                        model_p=model_p,
                        # {} -> y_hat, q(y|x). Shape: None -> (B', H'), (B', H', V)
                        model_q=lambda: self.model_head.sample(
                            h,
                            horizon_target,
                            do_sample=do_sample,
                            top_k=top_k,
                        ),
                        sample_fn=lambda p: sample_topk(p, top_k=top_k).squeeze(-1),
                    )  # (B', H') -- H' <= H_tgt if not all tokens are accepted

                    accept_rate_metrics["tokens_proposed"] += horizon_target
                    accept_rate_metrics["tokens_accepted"] += y_hat.size(1) - 1

                else:
                    y_hat, _ = self.model_head.sample(
                        h,
                        horizon_target,
                        do_sample=do_sample,
                        top_k=top_k,
                    )  # (B', H_tgt)

                horizon_prime = y_hat.size(1)

                # Append new tokens
                time_step_abs = x.size(1) + time_step
                y_out[
                    mask_active,
                    time_step_abs : time_step_abs + horizon_prime,
                ] = y_hat

                time_step += horizon_prime

        # Keep only the new tokens if requested
        if return_new_tokens:
            y_out = y_out[:, x.size(1) :]

        # Replace stop tokens with padding
        if stop_token is not None:
            y_out[y_out == temp_token] = stop_token
            stop_mask = (y_out == stop_token).float()  # (B, T_out)
            y_out[torch.cumsum(stop_mask, dim=1) >= 1] = stop_token

        return y_out, accept_rate_metrics

    def _get_tgt_loss(
        self,
        input_ids: torch.Tensor,
        last_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Get the target loss.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (B, T).
            last_hidden_state (torch.Tensor): Input tensor of shape (B, T, V).

        Returns:
            torch.Tensor: Target loss of shape (B, T).
        """
        py_x_tilde = self.tgt_model_head(last_hidden_state)  # (B, T, V)
        targets = get_windowed_input_ids(input_ids, horizon=1)  # (B, T-1, 1)
        # (B, T, V) -> (B, T, 1) -> (B, T)
        return torch.gather(
            -torch.log_softmax(py_x_tilde, dim=-1)[:, :-1],  # (B, T-1, V)
            dim=-1,
            index=targets,
        ).squeeze(-1)

    def _get_draft_loss(
        self,
        last_hidden_state: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ):
        """Compute negative log likelihood loss.

        Args:
            last_hidden_state (torch.Tensor): Last hidden state of the model. Shape (B, T, D)
            targets (torch.Tensor): Targets for the model. Shape (B, T, H)

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        p_tilde, p_tilde_scale_factors, norm_const, norm_const_scale_factors = (
            self.model_head.evaluate_at_points_and_get_norm_consts(
                last_hidden_state, targets
            )
        )  # (B, T-H)

        # Health checks
        # 1. Ensure no NaNs
        assert not torch.isnan(p_tilde).any(), "p_tilde NaN"
        assert not torch.isnan(norm_const).any(), "norm_const NaN"
        # 2. Ensure p_tilde < norm_const (if no scale factors)
        if len(p_tilde_scale_factors) == 0 and len(norm_const_scale_factors) == 0:
            if (p_tilde > norm_const).any():
                print("p_tilde >= norm_const")
                print("p_tilde:", p_tilde)
                print("norm_const:", norm_const)

            if not (p_tilde <= norm_const).all():
                print("p_tilde > norm_const")
            assert (p_tilde <= norm_const).all(), "p_tilde <= norm_const"

        loss = (
            -torch.log(p_tilde)  # (B, T')
            + torch.log(norm_const)  # (B, T')
            # Contraction Stability Scale Factors
            - sum([torch.log(z) for z in p_tilde_scale_factors])  # (B, T')
            + sum([torch.log(z) for z in norm_const_scale_factors])
        )  # (B, T-H)

        # Loss validation
        if (loss < 0).any():
            print("Detect negative loss")

        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        # NOTE: needed for compatibility with Trainer
        labels: torch.Tensor,
        attention_mask=None,
        horizon: Optional[int] = None,
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

        batch_size, _ = input_ids.size()
        horizon = self._get_horizon(horizon)

        last_hidden_state = self.get_attn_last_hidden_state(
            input_ids, attention_mask=attention_mask
        )
        targets = get_windowed_input_ids(input_ids, horizon=horizon).reshape(
            batch_size, -1, horizon
        )  # (B, T-H, H)

        assert targets.size(1) >= horizon, "Invalid targets"

        last_hidden_state_ds = last_hidden_state[:, :-horizon]  # (B, T-H, D)
        targets_ds = targets  # (B, T-H, H)
        if self.use_memory_efficient_loss:
            # Downsample hidden states and targets
            # (B, T-H // H, D), (B, T-H // H, H)
            shift = torch.randint(0, horizon, (1,)).item()
            last_hidden_state_ds = last_hidden_state_ds[:, shift::horizon]
            targets_ds = targets[:, shift::horizon]

        if hasattr(self.model_head, "compute_loss"):
            draft_loss = self.model_head.compute_loss(
                x=last_hidden_state_ds.reshape(-1, self.n_embd),
                y=targets_ds.reshape(-1, horizon),
            ).reshape(
                batch_size, -1
            )  # (B, T-H // H)
        else:
            draft_loss = self._get_draft_loss(
                last_hidden_state=last_hidden_state_ds,
                targets=targets_ds,
            )  # (B, T*)

        # Train loss
        loss_tot = draft_loss.sum(-1)
        if self.loss_mode == "joint":
            # Joint loss
            tgt_log_probs = self.tgt_model_head(
                last_hidden_state[:, :-1]
            )  # (B, T-1, V)
            tgt_gt = get_windowed_input_ids(
                input_ids,
                horizon=1,
            ).squeeze(
                -1
            )  # (B, T-1)
            tgt_loss = torch.nn.functional.cross_entropy(
                tgt_log_probs.view(-1, self.vocab_size),
                tgt_gt.view(-1),
                reduction="none",
            ).reshape(batch_size, -1)
            loss_tot = loss_tot + self.joint_loss_lambda * tgt_loss.sum(-1)  # (B,)

        # NLL computation requires only each horizon-th element
        nll = draft_loss if self.use_memory_efficient_loss else draft_loss[:, ::horizon]
        return {
            "loss": reduce_fn(loss_tot),
            "nll": reduce_fn(nll.sum(dim=-1)),
            "loss_scale": torch.tensor(1 / self.rank).to(draft_loss.device),
        }
