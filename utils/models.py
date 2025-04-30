import torch
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.models._tjd import TJD, TJDConfig
from tjdnet.models.tjdhf import TJDHuggingFace


def create_model(
    rank,
    horizon,
    hidden_dim,
    use_memory_efficient_loss=False,
    model="meta-llama/Llama-2-7b-chat-hf",
    model_head="cp",
    **kwargs,
):
    return lambda: TJDHuggingFace(
        TJDConfig(
            base_dist=BaseDistConfig(
                vocab_size=-1,  # will be set by tjd
                horizon=horizon,
                rank=rank,
                param_net=TensorParamNetConfig(hidden_dim=hidden_dim),
            ),
            model_head=model_head,
            auto_model_kwargs={"pretrained_model_name_or_path": model},
            use_memory_efficient_loss=use_memory_efficient_loss,
            **kwargs,
        ),
    )


def create_model_gpt_fn(
    rank,
    horizon,
    hidden_dim,
    model_head="cp",
    auto_model_kwargs={"pretrained_model_name_or_path": "gpt2"},
    **kwargs,
):
    return lambda: TJDHuggingFace(
        TJDConfig(
            base_dist=BaseDistConfig(
                vocab_size=-1,  # will be set by tjd
                rank=rank,
                horizon=horizon,
                param_net=TensorParamNetConfig(hidden_dim=hidden_dim),
            ),
            model_head=model_head,
            auto_model_kwargs=auto_model_kwargs,
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
