from typing import Any
import torch.nn as nn
from .TJDLayer import TJDLayer
from .RepNet import RepNet


class TJDNet(RepNet):
    def __init__(
        self,
        model: nn.Module,
        emb_size: int = 32,
        rank: int = 32,
        vocab_size: int = 128,
        *args,
        **kwargs
    ):
        """Tensor Train Joint Distribution Network"""

        def replacement_func(model: nn.Module) -> nn.Module:
            return TJDLayer(emb_size=emb_size, rank=rank, vocab_size=vocab_size)

        def condition_func(model: nn.Module, name: str) -> bool:
            return False

        super().__init__(model, condition_func, replacement_func, *args, **kwargs)
