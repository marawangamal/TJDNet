from typing import Literal, Optional, Tuple
from transformers import AutoConfig
import torch
from transformers import AutoModelForCausalLM
from transformers.utils import ModelOutput

from tjdnet.models.tjd import TJD, TJDConfig

from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

# Not all models have the same structure
EXCEPTIONS = {
    "gpt2": lambda m: (m.transformer, m.lm_head),
    "distilbert/distilgpt2": lambda m: (m.transformer, m.lm_head),
}


def get_lm_head_size(model_name: str) -> Tuple[int, int]:
    hf_config = AutoConfig.from_pretrained(model_name)
    vocab_size = hf_config.vocab_size
    if model_name in EXCEPTIONS:
        embedding_size = hf_config.n_embd
    elif hasattr(hf_config, "hidden_size"):
        embedding_size = hf_config.hidden_size
    else:
        raise ValueError(
            f"Model {model_name} is not supported for determining lm_head size."
        )
    return vocab_size, embedding_size


class TJDHuggingFace(TJD):
    def __init__(
        self,
        config: TJDConfig,
        auto_model_kwargs: dict,
        train_mode: Literal["full", "lora"] = "lora",
        lora_rank: int = 32,
        **kwargs,
    ):

        # 1. Initialize config
        self.hf_lora_rank = lora_rank
        self.hf_auto_model_kwargs = auto_model_kwargs
        self.hf_train_mode = train_mode

        # 2. Override `config`
        vocab_size, embedding_size = get_lm_head_size(
            auto_model_kwargs["pretrained_model_name_or_path"]
        )
        config.model_head_config.param_net.in_dim = embedding_size
        config.model_head_config.vocab_size = vocab_size

        # 3. Initialize the backbone, lm_head, mhead_attn
        super().__init__(config)
        self.mhead_attn = None
        if self.hf_train_mode == "last":
            self.mhead_attn = torch.nn.MultiheadAttention(
                embed_dim=embedding_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )

        # Trainer compatibility
        self.gradient_checkpointing_enable = self.backbone.gradient_checkpointing_enable

        # Freeze backbone params if using LoRA
        if self.hf_train_mode == "lora":
            print("Trainable LoRA params:")
            for name, param in self.backbone.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False

        # Freeze lm_head
        if config.loss_mode == "draft":
            for name, param in self.lm_head.named_parameters():
                param.requires_grad = False

    def get_model(self):
        hfmodel = AutoModelForCausalLM.from_pretrained(**self.hf_auto_model_kwargs)
        if self.hf_auto_model_kwargs["pretrained_model_name_or_path"] in EXCEPTIONS:
            backbone, lm_head = EXCEPTIONS[
                self.hf_auto_model_kwargs["pretrained_model_name_or_path"]
            ](hfmodel)
        else:
            backbone, lm_head = hfmodel.model, hfmodel.lm_head

        # Apply LoRA to backbone
        if self.hf_train_mode == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=self.hf_lora_rank,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            backbone.add_adapter(peft_config, adapter_name="lora_1")
            # backbone = get_peft_model(backbone, peft_config)  # type: ignore

        self.config = hfmodel.config

        # Delete the original model to free memory
        del hfmodel
        torch.cuda.empty_cache()

        return backbone, lm_head

    def forward_backbone(self, *args, past_key_values=None, use_cache=False, **kwargs):
        """
        Forward pass of the backbone model with optional KV-caching.

        Args:
            past_key_values (tuple, optional): Cached key/value states for efficient generation.
            use_cache (bool, optional): Whether to use KV-caching. Defaults to False.
            *args, **kwargs: Other arguments passed to the transformer backbone.

        Returns:
            h_targ (torch.Tensor): Hidden state for use by target head.
            h_draft (torch.Tensor): Hidden state for use by draft head.
            past_key_values (tuple, optional): Updated cache if use_cache is True.
        """
        # Pass KV-caching args to the HuggingFace transformer
        transformer_outputs = self.backbone(
            *args, past_key_values=past_key_values, use_cache=use_cache, **kwargs
        )
        h_targ = transformer_outputs.last_hidden_state
        h_draft = h_targ

        if self.mhead_attn is not None:
            h_draft, _ = self.attn(
                query=h_targ,
                key=h_targ,
                value=h_targ,
                need_weights=False,
            )

        # Return past_key_values if present (for generation)
        if hasattr(transformer_outputs, "past_key_values"):
            return h_targ, h_draft, transformer_outputs.past_key_values
        return h_targ, h_draft, None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask=None,
        reduce="mean",
        **kwargs,
    ):
        # This is a workaround for the Trainer's forward method
        # which expects a forward method in the model.
        out_dict = super().forward(input_ids, labels, attention_mask, reduce, **kwargs)
        return ModelOutput(**out_dict)
