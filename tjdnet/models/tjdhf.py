from typing import Literal
import torch
from transformers import AutoModelForCausalLM

from tjdnet.models.tjd import TJD, TJDConfig

from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

# Not all models have the same structure
EXCEPTIONS = {
    "gpt2": lambda m: (m.transformer, m.lm_head),
    "distilbert/distilgpt2": lambda m: (m.transformer, m.lm_head),
}


class TJDHuggingFace(TJD):
    def __init__(
        self,
        config: TJDConfig,
        auto_model_kwargs: dict,
        train_mode: Literal["full", "lora"] = "lora",
        lora_rank: int = 32,
        **kwargs
    ):

        # Determine model_head dimensions
        temp_model = AutoModelForCausalLM.from_pretrained(**auto_model_kwargs)
        self.hf_lora_rank = lora_rank
        self.hf_auto_model_kwargs = auto_model_kwargs
        self.hf_train_mode = train_mode
        lm_head = temp_model.lm_head
        if hasattr(lm_head, "weight"):
            vocab_size = lm_head.weight.size(0)
            embedding_size = lm_head.weight.size(1)
        else:
            raise ValueError("lm_head does not have a weight attribute.")

        # Clear the temporary model to free memory
        del temp_model
        torch.cuda.empty_cache()

        config.model_head_config.param_net.in_dim = embedding_size
        config.model_head_config.vocab_size = vocab_size
        super().__init__(config)

        # Add attn layer
        # self.attn = torch.nn.MultiheadAttention(
        #     embed_dim=embedding_size,
        #     num_heads=8,
        #     dropout=0.1,
        #     batch_first=True,
        # )

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
        # if self.hf_train_mode == "lora":
        #     peft_config = LoraConfig(
        #         task_type=TaskType.FEATURE_EXTRACTION,
        #         inference_mode=False,
        #         r=self.hf_lora_rank,
        #         lora_alpha=32,
        #         lora_dropout=0.1,
        #     )
        #     backbone = get_peft_model(backbone, peft_config)  # type: ignore

        self.config = hfmodel.config

        # Delete the original model to free memory
        del hfmodel
        torch.cuda.empty_cache()

        return backbone, lm_head

    def forward_backbone(self, *args, **kwargs):
        transformer_outputs = self.backbone(*args, **kwargs)
        h_targ = transformer_outputs.last_hidden_state

        # # Pass throgh attn layer
        # h_draft, _ = self.attn(
        #     query=h_targ,
        #     key=h_targ,
        #     value=h_targ,
        #     need_weights=False,
        # )

        return h_targ, h_targ
