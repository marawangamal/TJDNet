from transformers import AutoModelForCausalLM

from tjdnet.models.tjd_v2 import TJD, TJDConfig

from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

# NOTE: Not all models have the same structure
EXCEPTIONS = {
    "gpt2": lambda m: (m.transformer, m.lm_head),
}


class TJDHuggingFace(TJD):
    def __init__(
        self, config: TJDConfig, auto_model_kwargs: dict, lora_rank: int = 32, **kwargs
    ):
        self.hfmodel = AutoModelForCausalLM.from_pretrained(**auto_model_kwargs)
        self.hf_lora_rank = lora_rank
        self.hf_auto_model_kwargs = auto_model_kwargs
        lm_head = self.hfmodel.lm_head
        if hasattr(lm_head, "weight"):
            vocab_size = lm_head.weight.size(0)
            embedding_size = lm_head.weight.size(1)
        else:
            raise ValueError("lm_head does not have a weight attribute.")
        config.model_head_config.param_net.in_dim = embedding_size
        config.model_head_config.vocab_size = vocab_size
        super().__init__(config)

    def get_model(self):

        if self.hf_auto_model_kwargs["pretrained_model_name_or_path"] in EXCEPTIONS:
            backbone, lm_head = EXCEPTIONS[
                self.hf_auto_model_kwargs["pretrained_model_name_or_path"]
            ](self.hfmodel)
        else:
            backbone, lm_head = self.hfmodel.model, self.hfmodel.lm_head

        # Apply LoRA to backbone
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=self.hf_lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        backbone = get_peft_model(backbone, peft_config)  # type: ignore
        return backbone, lm_head
