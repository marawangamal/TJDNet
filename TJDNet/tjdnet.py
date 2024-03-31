from typing import Any, Callable
import torch.nn as nn
import torch
from TJDNet import TJDLayer
from RepNet import RepNet


class TJDNet(RepNet):
    def __init__(
        self,
        model: nn.Module,
        emb_size,
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
