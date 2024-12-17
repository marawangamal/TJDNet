from transformers import AutoModelForCausalLM

from models._tjd import TJD


class TJDLLAMA(TJD):
    def __init__(
        self,
        model_head: str = "base",
        vocab_size: int = 32000,
        n_embd: int = 5120,
        rank: int = 2,
        horizon: int = 8,
        positivity_func: str = "exp",
        freeze_base_model: bool = False,
        **kwargs,
    ):
        super().__init__(
            n_embd,
            vocab_size,
            rank=rank,
            horizon=horizon,
            model_head=model_head,
            positivity_func=positivity_func,
        )

        # NOTE: Unfreezing atleast the last layer is required for proper training
        if freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False
        for param in self.model.transformer.h[-1].mlp.parameters():
            param.requires_grad = True

    # TODO: use attention_mask
    def get_last_hidden_state(self, input_ids, attention_mask=None):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return transformer_outputs.last_hidden_state

    def get_model(self, **model_kwargs):
        return AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
