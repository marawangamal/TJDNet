from transformers import AutoModelForCausalLM

from tjdnet.models._tjd import TJD, TJDConfig


# NOTE: Not all models have the same structure
EXCEPTIONS = {
    "gpt2": lambda m: (m.transformer, m.lm_head),
}


class TJDHuggingFace(TJD):
    def __init__(self, config: TJDConfig, **kwargs):

        model = AutoModelForCausalLM.from_pretrained(**config.auto_model_kwargs)
        lm_head = model.lm_head
        if hasattr(lm_head, "weight"):
            vocab_size = lm_head.weight.size(0)
            embedding_size = lm_head.weight.size(1)
        else:
            raise ValueError("lm_head does not have a weight attribute.")

        config.base_dist.param_net.in_dim = embedding_size
        config.base_dist.vocab_size = vocab_size

        super().__init__(config)

    def get_last_hidden_state(self, input_ids, attention_mask=None):
        transformer_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = transformer_outputs.last_hidden_state
        return last_hidden_state

    def get_model(self, **auto_model_kwargs):
        model = AutoModelForCausalLM.from_pretrained(**auto_model_kwargs)
        if auto_model_kwargs["pretrained_model_name_or_path"] in EXCEPTIONS:
            return EXCEPTIONS[auto_model_kwargs["pretrained_model_name_or_path"]](model)
        return model.model, model.lm_head
