import torch
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.models._tjd import TJD, TJDConfig
from tjdnet.models.tjdgpt2 import TJDGPT2
from tjdnet.models.tjdllama import TJDLLAMA


def create_model_llama_fn(
    rank,
    horizon,
    model_head="cp",
    model_kwargs={"hf_model_name": "meta-llama/Llama-2-7b-chat-hf"},
    vocab_size=32000,
    param_net_config={
        "hidden_dim": 32000,
        "use_decoder": False,
    },
    **kwargs,
):
    return lambda: TJDLLAMA(
        TJDConfig(
            base_dist=BaseDistConfig(
                vocab_size=vocab_size,
                horizon=horizon,
                rank=rank,
                param_net=TensorParamNetConfig(**param_net_config),
            ),
            model_head=model_head,
            model_kwargs=model_kwargs,
            **kwargs,
        ),
    )


def create_model_gpt_fn(
    rank,
    horizon,
    model_head="cp",
    vocab_size=768,
    param_net_config={
        "hidden_dim": 768,  # should be vocab_size for base
        "use_decoder": True,
    },
    **kwargs,
):
    return lambda: TJDGPT2(
        TJDConfig(
            base_dist=BaseDistConfig(
                vocab_size=vocab_size,
                rank=rank,
                horizon=horizon,
                param_net=TensorParamNetConfig(**param_net_config),
            ),
            model_head=model_head,
            **kwargs,
        )
    )


def train_forward(
    model: TJD,
    input_ids: torch.Tensor,
):
    """Forward pass for training mode."""
    # Forward pass
    outputs = model.forward(input_ids, labels=input_ids)
    loss = outputs["loss"]
    # backward pass
    loss.backward()
    return loss
