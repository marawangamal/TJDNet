from transformers import AutoModelForCausalLM

from tjdnet.models.tjd_v2 import TJD, TJDConfig


# NOTE: Not all models have the same structure
EXCEPTIONS = {
    "gpt2": lambda m: (m.transformer, m.lm_head),
}


class TJDHuggingFace(TJD):
    def __init__(self, config: TJDConfig, auto_model_kwargs: dict, **kwargs):
        self.hfmodel = AutoModelForCausalLM.from_pretrained(**auto_model_kwargs)
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
        return self.hfmodel.model, self.hfmodel.lm_head
