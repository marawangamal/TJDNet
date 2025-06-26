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


class TJDSimple(nn.Module):
    """Super minimal TJD model combining TJD and TJDHF functionality."""

    def __init__(self, config: TJDSimpleConfig):
        super().__init__()
        self.config = config

        # Load HuggingFace model
        self.hf_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.backbone = self.hf_model.transformer
        self.lm_head = self.hf_model.lm_head

        # Get model dimensions
        hf_config = AutoConfig.from_pretrained(config.model_name)
        self.embedding_dim = hf_config.n_embd
        self.vocab_size = hf_config.vocab_size

        # Create distribution head
        dist_config = BaseDistConfig(
            vocab_size=self.vocab_size,
            horizon=config.horizon,
            rank=config.rank,
            embedding_dim=self.embedding_dim,
            positivity_func=config.positivity_func,
        )
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
            self.backbone.add_adapter(peft_config, adapter_name="lora_1")

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
        **kwargs
    ) -> ModelOutput:
        """Forward pass for training."""
        # Get hidden states
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_states = outputs.last_hidden_state

        # Compute loss if labels provided
        if labels is not None:
            # Prepare targets first to get the correct batch size
            y = self._prepare_targets(input_ids)  # (B*(T-H), H)

            # Reshape hidden states to match target batch size
            _, _, D = hidden_states.shape
            H = self.config.horizon
            # Remove last H positions since we can't predict them
            x = hidden_states[:, :-H, :]  # (B, T-H, D)
            x = x.reshape(-1, D)  # (B*(T-H), D)

            loss = self.dist_head(x, y).mean()
            return ModelOutput(**{"loss": loss, "nll": -1})

        return ModelOutput(**{"hidden_states": hidden_states})

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        generated = input_ids.clone()
        B = input_ids.size(0)
        H = horizon if horizon is not None else self.config.horizon
        device = input_ids.device

        for _ in range(max_new_tokens):
            outputs = self.backbone(input_ids=generated, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            sampled, _ = self.dist_head.sample(
                hidden_states[:, -1:],
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
