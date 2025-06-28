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


class MultiTokenLlamaConfig(PretrainedConfig):
    model_type = "multi_token_llama"

    def __init__(
        self,
        model_name: str = "gpt2",
        horizon: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.horizon = horizon


class MultiTokenLlama(PreTrainedModel, GenerationMixin):
    config_class = MultiTokenLlamaConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MultiTokenLlamaConfig):
        super().__init__(config)
        self.backbone, pretrained_lm_head = get_backbone(
            config.model_name, pretrained=True
        )
        self.vocab_size, self.embedding_dim = get_model_dims(config.model_name)
        self.horizon = config.horizon
        self.heads = nn.ModuleList(
            [
                nn.Linear(
                    self.embedding_dim,
                    self.vocab_size,
                    bias=pretrained_lm_head.bias is not None,
                )
                for _ in range(self.horizon)
            ]
        )
        # Initialize first head with pretrained lm_head weights
        with torch.no_grad():
            self.heads[0].weight.copy_(pretrained_lm_head.weight)
            if self.heads[0].bias is not None and pretrained_lm_head.bias is not None:
                self.heads[0].bias.copy_(pretrained_lm_head.bias)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def get_output_embeddings(self):
        # Return the first head as the output embedding (for generation)
        return self.heads[0]

    def get_input_embeddings(self):
        # Try to get input embeddings from the backbone
        if hasattr(self.backbone, "wte"):
            return self.backbone.wte
        elif hasattr(self.backbone, "embed_tokens"):
            return self.backbone.embed_tokens
        else:
            raise NotImplementedError("Input embeddings not found in backbone.")

    def set_output_embeddings(self, new_embeddings):
        self.heads[0] = new_embeddings

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

    def forward(self, input_ids, labels=None, **kwargs):
        # Get hidden states from model
        outputs = self.backbone(input_ids=input_ids, **kwargs)
        hidden_states = outputs.last_hidden_state

        # Compute loss if labels provided
        if labels is not None:
            seq_len = input_ids.shape[1]
            if seq_len <= self.horizon:
                raise ValueError(
                    f"Input sequence length ({seq_len}) must be greater than horizon ({self.horizon}) for loss computation."
                )
            # Remove last H positions since we can't predict them
            x = hidden_states[:, : -self.horizon, :]  # (B, T-H, D)
            x = x.reshape(-1, self.embedding_dim)  # (B*(T-H), D)

            # Create targets: (B*(T-H), H)
            y = get_windowed_input_ids(input_ids, self.horizon).reshape(
                -1, self.horizon
            )

            # # Compute loss from each head
            # NOTE: this is incorrect but simpler for testing memory usage
            # total_loss = 0.0
            # for head in self.heads:
            #     logits = head(x)  # (B*(T-H), vocab_size)
            #     loss = nn.functional.cross_entropy(logits, y)
            #     total_loss += loss

            # Compute loss from each head
            logits = torch.stack([head(x) for head in self.heads], dim=1)  # (B', H, V)
            losses = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size), y.reshape(-1), reduction="none"
            )  # (B',)
            loss = losses.mean()
            return CausalLMOutput(loss=loss, logits=logits)

        # For inference: return logits from last position
        logits = self.heads[0](hidden_states[:, -1:, :])  # Use first head for inference
        return CausalLMOutput(logits=logits)
