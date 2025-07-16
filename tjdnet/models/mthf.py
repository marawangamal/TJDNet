from tjdnet.distributions import TJD_DISTS
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.types import ModelHeadType, PositivityFuncType
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from mtllama.utils import get_windowed_input_ids


def get_backbone(
    model_name: str, pretrained: bool = False
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Get a randomly initialized backbone of a HuggingFace model."""
    if pretrained:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        hf_model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_name)
        )

    if hasattr(hf_model, "transformer"):
        return hf_model.transformer, hf_model.lm_head
    elif hasattr(hf_model, "model"):  # e.g., Llama
        return hf_model.model, hf_model.lm_head
    else:
        raise ValueError(f"Cannot find transformer/model backbone in {type(hf_model)}")


def get_model_dims(model_name: str) -> tuple[int, int]:
    """Get vocabulary size and embedding dimension from model config."""
    hf_config = AutoModelForCausalLM.from_pretrained(model_name).config
    vocab_size = hf_config.vocab_size
    if hasattr(hf_config, "n_embd"):  # e.g., GPT
        embedding_size = hf_config.n_embd
    elif hasattr(hf_config, "hidden_size"):  # e.g., Llama
        embedding_size = hf_config.hidden_size
    else:
        raise ValueError(
            f"Model {model_name} is not supported for determining dimensions."
        )
    return vocab_size, embedding_size


class MultiTokenHFConfig(PretrainedConfig):
    model_type = "multi_token_llama"

    def __init__(
        self,
        model_name: str = "gpt2",
        horizon: int = 2,
        rank: int = 1,
        model_head: ModelHeadType = "cp",
        positivity_func: PositivityFuncType = "sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.horizon = horizon
        self.model_head = model_head
        self.rank = rank
        self.positivity_func = positivity_func


class MultiTokenHF(PreTrainedModel, GenerationMixin):
    config_class = MultiTokenHFConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MultiTokenHFConfig):
        super().__init__(config)
        self.backbone, _ = get_backbone(config.model_name, pretrained=False)
        self.vocab_size, self.embedding_dim = get_model_dims(config.model_name)
        self.horizon = config.horizon
        self.lm_head = TJD_DISTS[config.model_head](
            BaseDistConfig(
                vocab_size=self.vocab_size,
                horizon=self.horizon,
                embedding_dim=self.embedding_dim,
                rank=config.rank,
                positivity_func=config.positivity_func,
            )
        )

    def get_output_embeddings(self):
        return self.lm_head.get_output_embeddings()

    def get_input_embeddings(self):
        # Try to get input embeddings from the backbone
        if hasattr(self.backbone, "wte"):
            return self.backbone.wte
        elif hasattr(self.backbone, "embed_tokens"):
            return self.backbone.embed_tokens
        else:
            raise NotImplementedError("Input embeddings not found in backbone.")

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.set_output_embeddings(new_embeddings)

    def set_input_embeddings(self, new_embeddings):
        if hasattr(self.backbone, "wte"):
            self.backbone.wte = new_embeddings
        elif hasattr(self.backbone, "embed_tokens"):
            self.backbone.embed_tokens = new_embeddings
        else:
            raise NotImplementedError("Input embeddings not found in backbone.")

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Standard for decoder-only models: just return input_ids and any attention_mask
        model_inputs = {"input_ids": input_ids}
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            model_inputs["attention_mask"] = kwargs["attention_mask"]
        return model_inputs

    def adjust_logits_during_generation(self, logits, **kwargs):
        # No adjustment by default
        return logits

    def forward(self, input_ids, labels=None, use_memory_efficient_loss=True, **kwargs):
        # Get hidden states from model
        outputs = self.backbone(input_ids=input_ids, **kwargs)
        hidden_states = outputs.last_hidden_state  # (B, T-H, D)

        # Compute loss if labels provided
        if labels is not None:
            seq_len = input_ids.shape[1]
            if seq_len <= self.horizon:
                raise ValueError(
                    f"Input sequence length ({seq_len}) must be greater than horizon ({self.horizon}) for loss computation."
                )
            # Remove last H positions since we can't predict them
            x = hidden_states[:, : -self.horizon, :]  # (B, T-H, D)

            # Create targets: (B*(T-H), H)
            y = get_windowed_input_ids(input_ids, self.horizon)
            if use_memory_efficient_loss:
                shift = torch.randint(0, self.horizon, (1,)).item()
                x = x[:, shift :: self.horizon]
                y = y[:, shift :: self.horizon]

            # Merge batch and sequence dims
            x = x.reshape(-1, self.embedding_dim)  # (B*(T-H), D)
            y = y.reshape(-1)  # (B*(T-H),)

            # # Compute loss from each head
            # NOTE: this is incorrect but simpler for testing memory usage
            # total_loss = 0.0
            # for head in self.heads:
            #     logits = head(x)  # (B*(T-H), vocab_size)
            #     loss = nn.functional.cross_entropy(logits, y)
            #     total_loss += loss

            output = self.lm_head(x, y)
            loss = output.loss.mean()
            logits = output.logits
            # Compute loss from each head
            # logits = torch.stack([head(x) for head in self.heads], dim=1)  # (B', H, V)
            # losses = nn.functional.cross_entropy(
            #     logits.reshape(-1, self.vocab_size), y.reshape(-1), reduction="none"
            # )  # (B',)
            # loss = losses.mean()
            return CausalLMOutput(loss=loss, logits=logits)

        # For inference: return logits from last position
        logits = self.lm_head(hidden_states[:, -1:, :])
        return CausalLMOutput(logits=logits)
