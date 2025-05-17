import torch
from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.models.tjd import TJD, TJDConfig
from tjdnet.models.tjdhf import TJDHuggingFace
from tjdnet.utils import mem_check


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
            model_head=model_head,
            model_head_config=BaseDistConfig(
                vocab_size=-1,  # will be set by tjd
                horizon=horizon,
                rank=rank,
                param_net=TensorParamNetConfig(hidden_dim=hidden_dim),
            ),
            use_memory_efficient_loss=use_memory_efficient_loss,
        ),
        auto_model_kwargs={"pretrained_model_name_or_path": model},
    )


def train_forward(
    model: TJD,
    input_ids: torch.Tensor,
):
    """Forward pass for training mode."""
    # Forward pass
    # mem_check("before model.forward")
    outputs = model.forward(input_ids, input_ids=input_ids)
    # mem_check("after model.forward")
    loss = outputs["loss"]

    # mem_check("before loss.backward")
    loss.backward()
    # mem_check("after loss.backward")
    return loss
