from typing import Literal
from transformers import AutoConfig
import torch
from transformers import AutoModelForCausalLM

from tjdnet.models.tjd import TJD, TJDConfig

from peft import LoraConfig, TaskType

# Not all models have the same structure
EXCEPTIONS = {
    "gpt2": lambda m: (m.transformer, m.lm_head),
    "distilbert/distilgpt2": lambda m: (m.transformer, m.lm_head),
}


def get_lm_head_size(model_name: str) -> tuple[int, int]:
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

        # Determine model_head dimensions
        vocab_size, embedding_size = get_lm_head_size(
            auto_model_kwargs["pretrained_model_name_or_path"]
        )

        self.hf_lora_rank = lora_rank
        self.hf_auto_model_kwargs = auto_model_kwargs
        self.hf_train_mode = train_mode

        config.model_head_config.param_net.in_dim = embedding_size
        config.model_head_config.vocab_size = vocab_size
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

        self.config = hfmodel.config

        # Delete the original model to free memory
        del hfmodel
        torch.cuda.empty_cache()

        return backbone, lm_head

    def forward_backbone(
        self, mode: Literal["draft", "target", "mixed"] = "mixed", **kwargs
    ):

        h_targ, h_draft = None, None

        # Last mode: can share hidden state for both target and draft
        if self.hf_train_mode == "last":
            transformer_outputs_shared = self.backbone(**kwargs)
            h_targ = transformer_outputs_shared.last_hidden_state
            h_draft = h_targ

        # Full mode: compute target and draft separately
        elif self.hf_train_mode == "lora":
            # Compute h_targ
            if mode in ["targ", "mixed"]:
                self.backbone.enable_adapters()
                transformer_outputs_targ = self.backbone(**kwargs)
                h_targ = transformer_outputs_targ.last_hidden_state

            # Compute h_draft
            if mode in ["draft", "mixed"]:
                # Disable adapter for draft mode
                self.backbone.disable_adapters()
                transformer_outputs_draft = self.backbone(**kwargs)
                h_draft = transformer_outputs_draft.last_hidden_state

        # (Optional) final attn layer for last mode
        if h_draft is not None and self.mhead_attn is not None:
            h_draft, _ = self.mhead_attn(
                query=h_targ,
                key=h_targ,
                value=h_targ,
                need_weights=False,
            )

        return h_targ, h_draft
