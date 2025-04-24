from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Tuple, Type

from peft import LoraConfig, TaskType, get_peft_model  # type: ignore


import torch

from tjdnet.distributions._base import BaseDistConfig, BaseDistribution
from tjdnet.distributions.base import BaseDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.mps import MPSDist
from tjdnet.distributions.ucp import UCPDist
from tjdnet.distributions.umps import UMPSDist
from tjdnet.tensorops.common import get_windowed_input_ids
from tjdnet.utils import diagnose

DIST_MAP: Dict[str, Type[BaseDistribution]] = {
    "base": BaseDist,
    "cp": CPDist,
    "cpo": CPDist,
    "ucp": UCPDist,
    "mps": MPSDist,
    "umps": UMPSDist,
}


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
        model_kwargs (Dict): Additional arguments passed to the base model.

        # Training Configuration
        init_method (Literal["random", "pretrained"]): Model initialization method.
            - "random": Initialize weights randomly (default)
            - "pretrained": Load pretrained weights
        freeze_base_model (bool): If True, freezes the base model parameters.
        use_memory_efficient_loss (bool): If True, uses memory-efficient loss computation.

        # Numerical Parameters
        eps (float): Small value for numerical stability in computations.
    """

    # Core configuration
    base_dist: BaseDistConfig

    # Model architecture
    model_head: str = "base"
    model_kwargs: Dict = field(default_factory=dict)

    # Training configuration
    init_method: Literal["random", "pretrained"] = "random"
    train_mode: Literal["full", "last", "lora"] = "full"
    lora_rank: int = 512
    use_memory_efficient_loss: bool = False
    use_attn_layer: bool = False

    # Numerical parameters
    eps: float = 1e-9

    # Generation parameters
    eos_token_id: Optional[int] = None

    # Debugging
    gen_version: int = 1


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
        self.n_embd = config.base_dist.param_net.in_dim
        self.eps = config.eps
        self.gen_version = config.gen_version
        self.use_attn_layer = config.use_attn_layer
        self.use_memory_efficient_loss = config.use_memory_efficient_loss

        # DEBUG: LoraConfig
        if config.train_mode == "full":
            self.model = self.get_model(**config.model_kwargs)
        elif config.train_mode == "last":
            self.model = self.get_model(**config.model_kwargs)
            self.freeze_base_model()
        elif config.train_mode == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.model = get_peft_model(
                self.get_model(**config.model_kwargs), peft_config  # type: ignore
            )
        else:
            raise ValueError(f"Invalid train_mode: {config.train_mode}")

        self.model_head = DIST_MAP[config.model_head](config.base_dist)
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
            pt_weight, pt_bias = self.get_pretrained_lm_head_weights()
            self.model_head.init_params(pt_weight, pt_bias)

        # Trainer compatibility
        self.gradient_checkpointing_enable = self.model.gradient_checkpointing_enable

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
    def get_model(self, **kwargs) -> torch.nn.Module:
        """Get the torch model to be modified.

        Returns:
            torch.nn.Module: Model to be modified.
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

    def get_pretrained_lm_head_weights(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get the language model head weights + bias. Used for initializing the model head."""
        raise NotImplementedError(
            "get_pretrained_lm_head_weights must be implemented for pretrained init"
        )

    def freeze_base_model(self):
        """Freeze the base model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
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

        assert torch.all(input_ids >= 0), "Input tokens must be positive"
        assert stop_token is None or stop_token > 0, "Stop token must be positive"
        if attention_mask is not None:
            assert torch.all(
                torch.tensor(input_ids.shape == attention_mask.shape)
            ), "Shape mismatch between input_ids and attention_mask"

        horizon = self._get_horizon(horizon)
        batch_size = input_ids.size(0)
        device = input_ids.device

        temp_token = -100

        # Initialize output with input_ids
        output_seqs = torch.full(
            (batch_size, input_ids.size(1) + max_new_tokens),
            fill_value=temp_token,
            dtype=torch.long,
            device=device,
        )
        output_seqs[:, : input_ids.size(1)] = input_ids

        with torch.no_grad():
            for time_step in range(0, max_new_tokens, horizon):
                # Exit if all sequences are done
                active_mask = (
                    ~torch.any(output_seqs[:, input_ids.size(1) :] == stop_token, dim=1)
                    if stop_token is not None
                    else torch.ones(batch_size, device=device).bool()
                )  # (B,)
                if not active_mask.any():
                    break

                # Get currently active sequences
                active_seqs = output_seqs[
                    active_mask, : input_ids.size(1) + time_step
                ]  # (B_active, T_in + time_step)
                current_attention = (
                    extend_attn(attention_mask[active_mask], time_step)
                    if attention_mask is not None
                    else None
                )  # (B_active, T_in + time_step)

                # Generate next tokens
                hidden_state = self.get_attn_last_hidden_state(
                    input_ids=active_seqs, attention_mask=current_attention
                )

                horizon_curr = min(horizon, max_new_tokens - time_step)

                y_hat = self.model_head.sample(
                    hidden_state,
                    horizon_curr,
                    do_sample=do_sample,
                    top_k=top_k,
                )  # (B_active, horizon)

                # Append new tokens
                time_step_abs = input_ids.size(1) + time_step
                output_seqs[
                    active_mask,
                    time_step_abs : time_step_abs + horizon,
                ] = y_hat

        # Keep only the new tokens if requested
        if return_new_tokens:
            output_seqs = output_seqs[:, input_ids.size(1) :]

        # Replace stop tokens with padding
        if stop_token is not None:
            output_seqs[output_seqs == temp_token] = stop_token
            stop_mask = (output_seqs == stop_token).float()  # (B, T_out)
            output_seqs[torch.cumsum(stop_mask, dim=1) >= 1] = stop_token

        return output_seqs

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

        # Sequence length must be greater than horizon
        assert (
            input_ids.size(1) > self.horizon
        ), "Sequence length must be greater than horizon"

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
            last_hidden_state_ds = last_hidden_state_ds[:, ::horizon]
            targets_ds = targets[:, ::horizon]

        p_tilde, p_tilde_scale_factors, norm_const, norm_const_scale_factors = (
            self.model_head.evaluate_at_points_and_get_norm_consts(
                last_hidden_state_ds, targets_ds
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

        # assert (loss >= 0).all(), "Loss < 0"
        # diagnose(loss, "loss")
        # diagnose(p_tilde, "p_tilde")
        # diagnose(norm_const, "norm_const")
        # [
        #     diagnose(z, f"p_tilde_scale_factors::{i}")
        #     for i, z in enumerate(p_tilde_scale_factors)
        # ]
        # [
        #     diagnose(z, f"norm_const_scale_factors::{i}")
        #     for i, z in enumerate(norm_const_scale_factors)
        # ]

        # Train loss
        # NLL computation requires only each horizon-th element
        nll = loss if self.use_memory_efficient_loss else loss[:, ::horizon]
        reduct_fn = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x,
        }[reduce]
        return {
            "loss": reduct_fn(loss.sum(dim=-1)),
            "nll": reduct_fn(nll.sum(dim=-1)),
            "loss_scale": torch.tensor(1 / self.rank).to(loss.device),
        }
