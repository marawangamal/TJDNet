from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Tuple, Type

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

    # Generation parameters
    eos_token_id: Optional[int] = None

    # Debug
    fw_version: int = 1


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
        self.eps = config.eps
        self.use_attn_layer = config.use_attn_layer
        self.use_memory_efficient_loss = config.use_memory_efficient_loss
        self.use_speculative_sampling = config.use_speculative_sampling
        self.fw_version = config.fw_version

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

    # def get_pretrained_lm_head_weights(
    #     self,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     """Get the language model head weights + bias. Used for initializing the model head."""
    #     raise NotImplementedError(
    #         "get_pretrained_lm_head_weights must be implemented for pretrained init"
    #     )

    def freeze_base_model(self):
        """Freeze the base model."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    # TODO: use x instead of active_seqs
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

        temp_token = -100  # Temporary token for padding

        # Initialize output with input_ids
        output_seqs = torch.full(
            (batch_size, input_ids.size(1) + max_new_tokens),
            fill_value=temp_token,
            dtype=torch.long,
            device=device,
        )  # (B, T_in + T_out)
        output_seqs[:, : input_ids.size(1)] = input_ids

        accept_rate_metrics = {
            "tokens_proposed": 0,
            "tokens_accepted": 0,
            "num_tokens_generated": 0,
            "num_speculative_tokens_accepted": 0,
        }

        with torch.no_grad():
            time_step = 0
            # for time_step in range(0, max_new_tokens, horizon):
            while time_step < max_new_tokens:
                time_step_prime = input_ids.size(1) + time_step
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
                    active_mask, :time_step_prime
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

                horizon_target = min(
                    horizon, max_new_tokens - time_step
                )  # speculative heads may not reach target horizon

                def model_p(y: torch.Tensor):
                    # Signature: model_p: y -> p(y|x)
                    # Shape of y: (B_active, H')
                    x = active_seqs  # (B_active, T_in + time_step)
                    h = self.get_attn_last_hidden_state(
                        input_ids=torch.cat((x, y), dim=1),
                        attention_mask=current_attention,
                    )  # (B_active, T_in + time_step + H', D)
                    py_x = self.tgt_model_head(h)  # (B_active, t' + H', V)
                    # TODO: maybe should abasorb this into the tgt_model_head
                    return torch.softmax(py_x[:, time_step_prime:], dim=-1)

                if self.use_speculative_sampling:
                    y_hat = spec_sample_v2(
                        # model_p: y -> p(y|x)
                        model_p=model_p,
                        # {} -> y_hat, q(y|x)
                        model_q=lambda: self.model_head.sample(
                            hidden_state,
                            horizon_target,
                            do_sample=do_sample,
                            top_k=top_k,
                        ),
                        sample_fn=lambda p: sample_topk(p, top_k=top_k).squeeze(-1),
                    )  # (B', H') -- H' <= H_tgt

                    accept_rate_metrics["tokens_proposed"] += horizon_target
                    accept_rate_metrics["tokens_accepted"] += y_hat.size(1)

                else:
                    y_hat, _ = self.model_head.sample(
                        hidden_state,
                        horizon_target,
                        do_sample=do_sample,
                        top_k=top_k,
                    )  # (B_active, horizon_actual)
                horizon_actual = y_hat.size(1)

                accept_rate_metrics["num_tokens_generated"] += y_hat.size(1)
                accept_rate_metrics["num_speculative_tokens_accepted"] += (
                    y_hat.size(1) - 1 if self.use_speculative_sampling else 0
                )

                # Append new tokens
                time_step_abs = input_ids.size(1) + time_step
                output_seqs[
                    active_mask,
                    time_step_abs : time_step_abs + horizon_actual,
                ] = y_hat

                time_step += horizon_actual

        # Keep only the new tokens if requested
        if return_new_tokens:
            output_seqs = output_seqs[:, input_ids.size(1) :]

        # Replace stop tokens with padding
        if stop_token is not None:
            output_seqs[output_seqs == temp_token] = stop_token
            stop_mask = (output_seqs == stop_token).float()  # (B, T_out)
            output_seqs[torch.cumsum(stop_mask, dim=1) >= 1] = stop_token

        return output_seqs, accept_rate_metrics

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

    def _compute_loss(
        self,
        last_hidden_state: torch.Tensor,
        targets: torch.Tensor,
        horizon: int,
        reduce: str = "mean",
    ):
        """Compute the loss.

        Args:
            last_hidden_state (torch.Tensor): Last hidden state of the model. Shape (B, T, D)
            targets (torch.Tensor): Targets for the model. Shape (B, T-H, H)
            horizon (int): Horizon for the model.
            reduce (str, optional): Defaults to "mean".

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - loss (torch.Tensor): Reduced loss value of shape (B,)
                - nll (torch.Tensor): Reduced negative log likelihood of shape (B,)
                - loss_scale (torch.Tensor): Loss scaling factor, scalar tensor of value 1/rank
        """

        if hasattr(self.model_head, "compute_loss"):
            pass

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

        if self.loss_mode == "joint":
            # loss_tgt = self._get_tgt_loss(
            #     last_hidden_state=last_hidden_state, input_ids=input_ids
            # )
            # loss = loss + loss_tgt
            raise NotImplementedError("Joint loss not implemented.")

        # Loss validation
        if (loss < 0).any():
            print("Detect negative loss")

        return loss

        # Train loss
        # NLL computation requires only each horizon-th element
        # nll = loss if self.use_memory_efficient_loss else loss[:, ::horizon]
        # reduct_fn = {
        #     "mean": torch.mean,
        #     "sum": torch.sum,
        #     "none": lambda x: x,
        # }[reduce]
        # return {
        #     "loss": reduct_fn(loss.sum(dim=-1)),
        #     "nll": reduct_fn(nll.sum(dim=-1)),
        #     "loss_scale": torch.tensor(1 / self.rank).to(loss.device),
        # }

    def compute_loss_v2(
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

        if self.loss_mode == "joint":
            # loss_tgt = self._get_tgt_loss(
            #     last_hidden_state=last_hidden_state, input_ids=input_ids
            # )
            # loss = loss + loss_tgt
            raise NotImplementedError("Joint loss not implemented.")

        # Loss validation
        if (loss < 0).any():
            print("Detect negative loss")

        return loss

    def forward_v2(
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

        reduce_func = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x,
        }[reduce]

        mem_check("before get_attn_last_hidden_state")

        # (B, T-H, D)
        last_hidden_state = self.get_attn_last_hidden_state(
            input_ids, attention_mask=attention_mask
        )[:, :-horizon]
        targets = get_windowed_input_ids(input_ids, horizon=horizon).reshape(
            batch_size, -1, horizon
        )  # (B, T-H, H)

        mem_check("after get_attn_last_hidden_state")

        assert targets.size(1) >= horizon, "Invalid targets"

        num_shifts = 1
        if self.use_memory_efficient_loss:
            num_shifts = horizon

        for shift in range(num_shifts):
            # Downsample hidden states and targets
            # (B, T-H, D) -> (B, T-H // H, D)
            last_hidden_state_ds = last_hidden_state[:, shift::horizon]
            targets_ds = targets[:, shift::horizon]  # (B, T-H // H, H)
            mem_check("before _compute_loss")
            if shift == 0:
                loss = self._compute_loss(
                    last_hidden_state=last_hidden_state_ds,
                    targets=targets_ds,
                    horizon=horizon,
                )  # (B, T-H // H)
                loss_dict = {
                    "loss": loss.sum(dim=-1),
                    "nll": loss.sum(dim=-1),
                    "loss_scale": torch.tensor(1 / self.rank).to(loss.device),
                }
            else:
                loss_i = self._compute_loss(
                    last_hidden_state=last_hidden_state_ds,
                    targets=targets_ds,
                    horizon=horizon,
                )
                loss_dict["loss"] += reduce_func(loss.sum(dim=-1))
                # loss_dict["nll"] += loss_dict_i["nll"]
                # loss_dict["loss_scale"] += loss_dict_i["loss_scale"]

            mem_check("after _compute_loss")
            # Return means
            loss_dict["loss"] = loss_dict["loss"] / num_shifts
            loss_dict["nll"] = loss_dict["nll"] / num_shifts
            loss_dict["loss_scale"] = loss_dict["loss_scale"] / num_shifts

        return loss_dict

    def forward_v1(
        self,
        input_ids: torch.Tensor,
        # NOTE: needed for compatibility with Trainer
        labels: torch.Tensor,
        attention_mask=None,
        horizon: Optional[int] = None,
        reduce="mean",
        shift=0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (B, T)
            labels (torch.Tensor): Tensor of shape (B, T)
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.
            reduce (str, optional): Reduction method. Defaults to "mean".
            shift (int, optional): Shift for downsampling. Defaults to 0.
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
            },
            {
                "test": lambda: shift < self.horizon,
                "msg": "Shift must be less than horizon",
            },
        ]
        for check in input_validation_checks:
            assert check["test"](), check["msg"]
        # ====

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
            last_hidden_state_ds = last_hidden_state_ds[:, shift::horizon]
            targets_ds = targets[:, shift::horizon]

        if hasattr(self.model_head, "compute_loss"):
            loss = self.model_head.compute_loss(
                x=last_hidden_state_ds.reshape(-1, self.n_embd),
                y=targets_ds.reshape(-1, horizon),
            ).reshape(
                batch_size, -1
            )  # (B, T-H // H)
        else:
            loss = self.compute_loss_v2(
                last_hidden_state=last_hidden_state_ds,
                targets=targets_ds,
            )  # (B, T*)

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

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        return {
            1: self.forward_v1,
            2: self.forward_v2,
        }[
            self.fw_version
        ](input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
