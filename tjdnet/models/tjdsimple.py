import torch
import torch.nn as nn
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from typing import Optional, Literal

from tjdnet.distributions import TJD_DISTS
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.tensorops.common import get_windowed_input_ids_v2
from tjdnet.types import PositivityFuncType, ModelHeadType


@dataclass
class TJDSimpleConfig:
    """Minimal config for TJDSimple model."""

    model_name: str = "gpt2"
    model_head: ModelHeadType = "multihead"
    horizon: int = 3
    rank: int = 4
    train_mode: Literal["full", "lora"] = "lora"
    lora_rank: int = 32
    positivity_func: PositivityFuncType = "exp"


@dataclass
class TJDGenerationConfig:
    """Minimal generation config."""

    max_new_tokens: int = 32
    do_sample: bool = False
    top_k: int = 1
    eos_token_id: Optional[int] = None
    positivity_func: PositivityFuncType = "exp"


def get_backbone(model_name: str) -> torch.nn.Module:
    """Get the backbone of a HuggingFace model."""
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    if hasattr(hf_model, "transformer"):
        return hf_model.transformer
    elif hasattr(hf_model, "model"):  # e.g., Llama
        return hf_model.model
    else:
        raise ValueError(f"Cannot find transformer/model backbone in {type(hf_model)}")


def get_lm_head_dims(model_name: str) -> tuple[int, int]:
    hf_config = AutoConfig.from_pretrained(model_name)
    vocab_size = hf_config.vocab_size
    if hasattr(hf_config, "n_embd"):  # e.g., GPT
        embedding_size = hf_config.n_embd
    elif hasattr(hf_config, "hidden_size"):  # e.g., Llama
        embedding_size = hf_config.hidden_size
    else:
        raise ValueError(
            f"Model {model_name} is not supported for determining lm_head size."
        )
    return vocab_size, embedding_size


class TJDSimple(nn.Module):
    """Super minimal TJD model combining TJD and TJDHF functionality."""

    def __init__(self, config: TJDSimpleConfig):
        super().__init__()
        self.config = config

        # Load HuggingFace model
        self.hf_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.backbone = get_backbone(config.model_name)
        self.lm_head = self.hf_model.lm_head

        # Get model dimensions
        self.vocab_size, self.embedding_dim = get_lm_head_dims(config.model_name)

        # Create distribution head
        dist_config = BaseDistConfig(
            vocab_size=self.vocab_size,
            horizon=config.horizon,
            rank=config.rank,
            embedding_dim=self.embedding_dim,
            positivity_func=config.positivity_func,
        )
        # try:
        #     self.dist_head = TJD_DISTS[config.model_head].from_pretrained(
        #         self.lm_head, dist_config
        #     )
        #     # set decoder requires_grad to False
        #     if hasattr(self.dist_head, "decoder"):
        #         for param in self.dist_head.decoder.parameters():
        #             param.requires_grad = False
        #         print("Dist head decoder frozen")
        #     print("Initialized dist head from pretrained")
        # except:
        #     self.dist_head = TJD_DISTS[config.model_head](dist_config)
        #     print("Initialized dist head from scratch")
        self.dist_head = TJD_DISTS[config.model_head](dist_config)

        # Apply LoRA if needed
        if config.train_mode == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.backbone.add_adapter(peft_config, adapter_name="lora_1")  # type: ignore

            # Freeze non-LoRA params
            for name, param in self.backbone.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False

        # Free original model
        del self.hf_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _prepare_targets(self, labels: torch.Tensor) -> torch.Tensor:
        # Use the same approach as in tjd.py
        y_true = get_windowed_input_ids_v2(labels, horizon=self.config.horizon)
        y_true = y_true.reshape(-1, self.config.horizon)  # (B*(T-H), H)
        return y_true

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_memory_efficient_loss: bool = True,
        **kwargs,
    ) -> ModelOutput:
        """Forward pass for training."""

        # Truncate to multiple of horizon
        # NOTE: Should handle partial targets in dist head in the future
        T = (input_ids.shape[1] // self.config.horizon) * self.config.horizon
        input_ids = input_ids[:, :T]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :T]
        if labels is not None:
            labels = labels[:, :T]

        # Get hidden states
        hidden_states = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        ).last_hidden_state  # (B, T, D)

        # Compute loss if labels provided
        if labels is not None:

            # Dims
            B, T, D = hidden_states.shape
            H = self.config.horizon

            # Prepare targets first to get the correct batch size
            y = self._prepare_targets(input_ids).reshape(B, -1, H)  # (B*(T-H), H)

            # Remove last H positions since we can't predict them
            z = hidden_states[:, :-H, :]  # (B, T-H, D)

            if use_memory_efficient_loss:
                shift = torch.randint(0, H, (1,)).item()
                z = z[:, shift::H]
                y = y[:, shift::H]

            z = z.reshape(-1, D)  # (B*T_eff, D)
            y = y.reshape(-1, H)  # (B*T_eff, H)
            # Check for batch size mismatch
            if z.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: z.shape[0]={z.shape[0]}, y.shape[0]={y.shape[0]}"
                )

            loss = self.dist_head(z, y).mean()
            return ModelOutput(
                **{"loss": loss, "nll": loss if use_memory_efficient_loss else -1}
            )

        return ModelOutput(**{"hidden_states": hidden_states})

    def forward_backward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        """Forward pass for training."""

        y = self._prepare_targets(input_ids)

        z = getattr(
            self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs),
            "last_hidden_state",
        )

        # truncate to multiple of horizon
        B, T, D = z.shape
        H = self.config.horizon
        # Truncate to the largest multiple of horizon that fits within T
        T_truncated = (T // H) * H
        if T_truncated < T:
            z = z[:, :T_truncated, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :T_truncated]
            if labels is not None:
                labels = labels[:, :T_truncated]

        z = z[:, :-H, :].reshape(-1, D)

        # Memory-efficient backward for MultiHeadDist
        if hasattr(self.dist_head, "forward_partial"):
            d = z.detach()
            d.requires_grad = True

            total_loss = 0.0
            for h in range(H):
                loss_i = self.dist_head.forward_partial(d, y, head_ids=[h]).mean()
                loss_i.backward()
                total_loss += loss_i.detach()
            z.backward(gradient=d.grad)
            return ModelOutput(**{"loss": total_loss, "nll": -1})

        # Fallback: normal backward
        loss = self.dist_head(z, y).mean()
        loss.backward()
        return ModelOutput(**{"loss": loss, "nll": -1})

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        generated = input_ids.clone()
        B = input_ids.size(0)
        H = horizon if horizon is not None else self.config.horizon
        device = input_ids.device

        for _ in range(max_new_tokens):
            outputs = self.backbone(input_ids=generated, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            sampled, _ = self.dist_head.sample(
                hidden_states[:, -1],
                lambda p: (
                    torch.multinomial(p, 1).squeeze(-1)
                    if do_sample
                    else torch.argmax(p, dim=-1)
                ),
                horizon=H,
            )
            generated = torch.cat([generated, sampled], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(B, 1, device=device)], dim=1
                )
            if eos_token_id is not None and (sampled == eos_token_id).any():
                break
        return generated
