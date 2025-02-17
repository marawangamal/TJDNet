from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional
from peft import LoraConfig, TaskType, get_peft_model


import torch

from distributions._base import BaseDistConfig
from distributions.base import BaseDist
from distributions.cp import CPDist
from distributions.full import FullDist
from distributions.mps import MPSDist
from distributions.umps import UMPSDist
from tensorops.common import get_windowed_input_ids
from utils.beam_search import beam_search, get_candidates

import line_profiler


DIST_MAP = {
    "full": FullDist,
    "cp": CPDist,
    "mps": MPSDist,
    "umps": UMPSDist,
    "base": BaseDist,
}


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

    # Numerical parameters
    eps: float = 1e-9

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
        self.rank = config.base_dist.rank
        self.horizon = config.base_dist.horizon
        self.vocab_size = config.base_dist.vocab_size
        self.n_embd = config.base_dist.param_net.in_dim
        self.eps = config.eps

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
                self.get_model(**config.model_kwargs), peft_config
            )
        else:
            raise ValueError(f"Invalid train_mode: {config.train_mode}")

        self.model_head = DIST_MAP[config.model_head](config.base_dist)
        self.use_memory_efficient_loss = config.use_memory_efficient_loss

        # Handle model initialization
        if config.init_method == "pretrained":
            weights = self.get_pretrained_lm_head_weights()
            self.model_head.init_params(weights)

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

    def get_pretrained_lm_head_weights(self) -> torch.Tensor:
        """Get the language model head weights. Used for initializing the model head."""
        raise NotImplementedError(
            "get_pretrained_lm_head_weights must be implemented for pretrained init"
        )

    def freeze_base_model(self):
        """Freeze the base model."""
        for param in self.model.parameters():
            param.requires_grad = False

    @line_profiler.profile
    def generate_v2(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 8,
        num_beams: int = 1,
        do_sample: bool = True,
        horizon: Optional[int] = None,
        top_k: int = 50,
        stop_token: Optional[int] = None,
        **kwargs,
    ):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.
            num_beams (int, optional): Number of beams. Defaults to 1.
            do_sample (bool, optional): Whether to sample. Defaults to False.
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.
            top_k (int, optional): Top k sampling. Defaults to 50.
            stop_token (Optional[int], optional): Stop token for generation. Defaults to None.

        Returns:
            torch.Tensor: Generated tokens of shape (B, `max_new_tokens`).
        """
        # Assert that batch size is 1
        assert input_ids.size(0) == 1, "Only batch size 1 is supported"
        input_ids_curr = input_ids.clone()
        dvc = input_ids.device
        with torch.no_grad():
            for t in range(max_new_tokens):
                last_hidden_state = self.get_last_hidden_state(input_ids)[
                    :, -1:, :
                ]  # forward pass
                ops_tensor = torch.tensor(
                    [-1],
                    device=dvc,
                )
                next_token_probs, _ = self.model_head.get_dist(
                    hidden_state=last_hidden_state,
                    ops=ops_tensor,
                )  # (V,)

                if do_sample:
                    # Apply top-k filtering
                    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k)
                    next_token = top_k_indices[torch.multinomial(top_k_probs, 1)]
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_probs, dim=-1).to(
                        input_ids_curr.device
                    )

                # Check for EOS token
                if next_token.item() == stop_token:
                    break

                # Append next token to input sequence
                input_ids_curr = torch.cat(
                    [input_ids_curr, next_token.reshape(1, 1)], dim=-1
                )

        return input_ids_curr

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 8,
        num_beams: int = 1,
        do_sample: bool = True,
        horizon: Optional[int] = None,
        top_k: int = 50,
        stop_token: Optional[int] = None,
        **kwargs,
    ):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.
            num_beams (int, optional): Number of beams. Defaults to 1.
            do_sample (bool, optional): Whether to sample. Defaults to False.
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.
            top_k (int, optional): Top k sampling. Defaults to 50.
            stop_token (Optional[int], optional): Stop token for generation. Defaults to None.

        Returns:
            torch.Tensor: Generated tokens of shape (B, `max_new_tokens`).
        """
        assert input_ids.size(0) == 1, "Only batch size 1 is supported"

        dvc = input_ids.device
        horizon = self._get_horizon(horizon)
        last_hidden_states = [
            self.get_last_hidden_state(input_ids)[:, -1:, :]
        ] * num_beams

        def expand_fn(beams):
            # beams: List[Tuple[seq: List[int], log_prob: float]]
            nonlocal last_hidden_states  # Allow modification of outer variable
            seqs, seq_log_probs = zip(*beams)  # Lists of shape (n_beams, T), (n_beams,)
            seq_log_probs = torch.tensor(seq_log_probs).to(dvc)

            # Since the tensorized model_head models the joint over the next `horizon` tokens,
            # we only need to do a forward pass (get_last_hidden_state) every `horizon` steps
            time_step = len(seqs[0])
            if time_step % horizon == 0 and time_step != 0:
                # print(f"[Hidden states] Time step: {time_step} (horizon: {horizon})")
                last_hidden_states = []
                seqs_tensor = torch.tensor(seqs).to(dvc)
                for sq in seqs_tensor:
                    inp = torch.cat([input_ids, sq.reshape(1, -1)], dim=1)
                    hidden = self.get_last_hidden_state(inp)[:, -1:, :]  # forward pass
                    last_hidden_states.append(hidden)

            next_token_probs = []
            for i_beam, seq in enumerate(seqs):
                sub_time_step = time_step % horizon
                sub_seq = seq[-sub_time_step:] if sub_time_step != 0 else []
                ops_tensor = torch.tensor(
                    sub_seq + [-1] + [-2] * (horizon - sub_time_step - 1),
                    device=dvc,
                )
                # NOTE: get_pos_params called many times with the same `hidden_state`
                probs_next, _ = self.model_head.get_dist(
                    hidden_state=last_hidden_states[i_beam],
                    ops=ops_tensor,
                    use_cache=False if sub_time_step == 0 else True,
                    save_cache=True,
                )  # (V,)
                assert (
                    len(probs_next.shape) == 1 and probs_next.size(0) == self.vocab_size
                ), "Invalid shape for probs_next"
                next_token_probs.append(probs_next)

            next_token_probs = torch.stack(next_token_probs)  # (n_beams, V)
            return get_candidates(
                seqs=seqs,
                seq_log_probs=seq_log_probs,
                next_token_probs=next_token_probs,
                num_beams=num_beams,
                do_sample=do_sample,
                top_k=top_k,
            )

        # Run beam search
        best_seq, _ = beam_search(
            expand_fn=expand_fn,
            initial_beam=[([], 0.0)],
            num_beams=num_beams,
            max_steps=max_new_tokens,
            stop_token=stop_token,
        )
        return torch.tensor(best_seq, device=dvc).reshape(1, -1)

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

        last_hidden_state = self.get_last_hidden_state(
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
            assert (p_tilde <= norm_const).all(), "p_tilde <= norm_const"

        loss = (
            -torch.log(p_tilde + self.eps)
            + torch.log(norm_const)
            # Contraction Stability Scale Factors
            - sum([torch.log(z) for z in p_tilde_scale_factors])
            + sum([torch.log(z) for z in norm_const_scale_factors])
        )  # (B, T-H)

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
