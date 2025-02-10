from typing import Optional
import torch
from models._tjd import TJD, TJDConfig

from transformers import AutoModelForCausalLM


class LLAMA(torch.nn.Module):
    def __init__(self, config: TJDConfig, **kwargs):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            low_cpu_mem_usage=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        # NOTE: needed for compatibility with Trainer
        labels: torch.Tensor,
        attention_mask=None,
        horizon: Optional[int] = None,
        reduce="mean",
        **kwargs,
    ):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = output.loss
        reduct_fn = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x,
        }[reduce]
        loss_reduction = reduct_fn(loss.sum(dim=-1))
        return {
            "loss": loss_reduction,
            "nll": loss_reduction,
            "loss_scale": torch.tensor(1).to(loss.device),
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 8,
        num_beams: int = 1,
        do_sample: bool = True,
        horizon: Optional[int] = None,
        top_k: int = 50,
        **kwargs,
    ):
        return self.model.generate(
            input_ids=input_ids,
            max_length=input_ids.size(1) + max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            top_k=top_k,
        )

    @property
    def param_dict(self):
        n_total_params = sum(p.numel() for p in self.model.parameters())
        n_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Get human readable format (in millions)
        n_trainable_params = f"{n_trainable_params / 1e6:.2f}M"
        n_total_params = f"{n_total_params / 1e6:.2f}M"

        return {
            "Trainable Params (M)": n_trainable_params,
            "Total Params (M)": n_total_params,
        }

    @property
    def device(self):
        return next(self.model.parameters()).device
