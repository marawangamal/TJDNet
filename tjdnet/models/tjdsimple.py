import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from dataclasses import dataclass
from typing import Optional, Dict, Literal

from tjdnet.distributions import TJD_DISTS
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.tensorops.common import get_windowed_input_ids_v2


@dataclass
class TJDSimpleConfig:
    """Minimal config for TJDSimple model."""

    model_name: str = "gpt2"
    model_head: str = "MultiHeadDist"
    horizon: int = 3
    rank: int = 4
    train_mode: Literal["full", "lora"] = "lora"
    lora_rank: int = 32


@dataclass
class TJDGenerationConfig:
    """Minimal generation config."""

    max_new_tokens: int = 32
    do_sample: bool = False
    top_k: int = 1
    eos_token_id: Optional[int] = None


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
        )
        self.dist_head = TJD_DISTS[config.model_head](dist_config)

        # Apply LoRA if needed
        if config.train_mode == "lora":
            from peft import LoraConfig, TaskType

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
        """Prepare targets for distribution head using the same approach as tjd.py.

        Args:
            labels: Shape (B, T) - batch of sequences

        Returns:
            Shape (B*(T-H), H) - flattened targets for each position
        """
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
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        # Get hidden states
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_states = outputs.last_hidden_state

        # Compute loss if labels provided
        if labels is not None:
            x = hidden_states.reshape(-1, self.embedding_dim)
            y = self._prepare_targets(labels)
            loss = self.dist_head(x, y).mean()
            return {"loss": loss}

        return {"hidden_states": hidden_states}

    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[TJDGenerationConfig] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate sequences."""
        if generation_config is None:
            generation_config = TJDGenerationConfig()

        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)

        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize output
        generated = input_ids.clone()

        for _ in range(generation_config.max_new_tokens):
            # Get hidden states
            outputs = self.backbone(input_ids=generated, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            # Sample next tokens
            sampled, _ = self.dist_head.sample(
                hidden_states[:, -1:],  # Use last token's hidden state
                lambda p: (
                    torch.argmax(p, dim=-1)
                    if not generation_config.do_sample
                    else torch.multinomial(p, 1).squeeze(-1)
                ),
                horizon=self.config.horizon,
            )

            # Append to generated sequence
            generated = torch.cat([generated, sampled], dim=1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 1, device=device)], dim=1
                )

            # Check for EOS
            if generation_config.eos_token_id is not None:
                if (sampled == generation_config.eos_token_id).any():
                    break

        return generated
