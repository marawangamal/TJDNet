from typing import Any, Callable, Optional, Tuple, Union, List
from transformers import (
    PreTrainedModel,
    GenerationConfig,
    StoppingCriteriaList,
    LogitsProcessorList,
)
import torch.nn as nn
import torch

from .RepNet import RepNet


class TJDOutput:
    def __init__(
        self,
        loss: Optional[torch.FloatTensor] = None,
        logits: Optional[torch.FloatTensor] = None,
    ):
        self.loss = loss
        self.logits = logits


class TJDNet(RepNet):
    def __init__(
        self,
        model: nn.Module,
        condition_func: Callable[[nn.Module, str], bool],
        replacement_func: Callable[[nn.Module], nn.Module],
        *args: Any,
        **kwargs: Any,
    ):
        """Abstract class for replacing modules in models.

        Args:
            model: base model to be modified
            condition_func: function that takes a module and returns a boolean (True if module should be replaced, False otherwise)
            replacement_func: function that takes a module and returns a new module

        """
        super().__init__(model, condition_func, replacement_func)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        loss = None
        if labels is not None:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.lm_head(shift_hidden, shift_labels)
        return TJDOutput(loss=loss)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer=None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.LongTensor:

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )
        max_length = kwargs.get("max_length")

        transformer_outputs = self.transformer(
            inputs,
            **model_kwargs,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        output = self.lm_head.get_preds(hidden_states, **kwargs)
        return output
