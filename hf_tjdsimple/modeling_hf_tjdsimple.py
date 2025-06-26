import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForCausalLM,
    AutoConfig,
)


class HFTJDSimpleConfig(PretrainedConfig):
    model_type = "hf_tjdsimple"

    def __init__(
        self, model_name="gpt2", horizon=2, rank=2, positivity_func="safe_exp", **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.horizon = horizon
        self.rank = rank
        self.positivity_func = positivity_func


class HFTJDSimple(PreTrainedModel):
    config_class = HFTJDSimpleConfig
    base_model_prefix = "hf_tjdsimple"

    def __init__(self, config):
        super().__init__(config)
        # Support both GPT-style and Llama-style models
        backbone_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        if hasattr(backbone_model, "transformer"):
            self.backbone = backbone_model.transformer
        elif hasattr(backbone_model, "model"):  # e.g., Llama
            self.backbone = backbone_model.model
        else:
            raise ValueError(
                f"Cannot find transformer/model backbone in {type(backbone_model)}"
            )
        self.lm_head = AutoModelForCausalLM.from_pretrained(config.model_name).lm_head
        self.horizon = config.horizon
        self.rank = config.rank
        self.positivity_func = config.positivity_func
        # For demonstration, use a simple linear head per horizon position
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.backbone.config.n_embd, self.backbone.config.vocab_size)
                for _ in range(self.horizon)
            ]
        )

    def forward(
        self,
        input_ids,
        labels=None,
        attention_mask=None,
        memory_efficient=False,
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_states = outputs.last_hidden_state
        B, T, D = hidden_states.shape
        H = self.horizon
        x = hidden_states[:, :-H, :].reshape(-1, D)  # (B*(T-H), D)
        if labels is not None:
            y = labels[:, -H:].reshape(-1)  # (B*H,)
            if memory_efficient:
                total_loss = 0.0
                for h, head in enumerate(self.heads):
                    # recompute hidden states for each head if needed for true efficiency
                    logits = head(x)
                    loss = nn.functional.cross_entropy(logits, y)
                    loss.backward()
                    total_loss += loss.detach()
                return {"loss": total_loss}
            else:
                logits = torch.stack(
                    [head(x) for head in self.heads], dim=1
                )  # (B*(T-H), H, vocab)
                logits = logits.mean(dim=1)  # simple average for demo
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                return {"loss": loss}
        return {"hidden_states": hidden_states}
