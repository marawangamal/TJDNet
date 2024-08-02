import torch
import torch.nn as nn


class BaseTensorizedDistribution(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseTensorizedDistribution, self).__init__()

    def select(self, y: torch.Tensor) -> torch.Tensor:
        # y: (seq_len,)
        # output: (1,)
        raise NotImplementedError

    def sample(self, max_len: int) -> torch.Tensor:
        # output: (max_len,)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, n_emb)
        # y: (batch_size, seq_len)
        # output: (1,)  -- loss function
        raise NotImplementedError
