import torch
from tjdnet.models._tjd import TJD, TJDConfig

from transformers import GPT2LMHeadModel


# TODO: instead of subclassing to make a new model, should just have TJD model accept 3 funcs in its args
# 1. get_last_hidden_state
# 2. get_pretrained_lm_head_weights
# 3. get_model
class TJDGPT2(TJD):
    def __init__(self, config: TJDConfig, **kwargs):
        config.base_dist.param_net.in_dim = 768
        # BUG: might be an issue since we add <pad> token to the vocab_size
        config.base_dist.vocab_size = 50257
        super().__init__(config)
        self.pretrained_weights = None

    def get_last_hidden_state(self, input_ids, attention_mask=None):
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = transformer_outputs.last_hidden_state
        del transformer_outputs
        torch.cuda.empty_cache()
        return last_hidden_state

    def get_pretrained_lm_head_weights(self):
        if self.pretrained_weights is None:
            raise ValueError("Pretrained weights not loaded.")
        return self.pretrained_weights

    def get_model(self, **model_kwargs):
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        transformer_model = model.transformer
        self.pretrained_weights = model.lm_head.weight
        del model
        torch.cuda.empty_cache()
        return transformer_model
