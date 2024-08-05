from typing import Tuple
import torch
import torch.nn as nn

from TJDNet.tensop import BaseTensorizedDistribution
from TJDNet.utils import umps_select_marginalize_batched


class MPSDist(nn.Module):
    def __init__(self, n_emb: int, n_vocab: int, rank: int = 2, *args, **kwargs):
        super(MPSDist, self).__init__()
        self.n_emb = n_emb
        self.n_vocab = n_vocab
        self.n_born_machine_params = n_vocab * rank * rank + 2 * rank
        self.w_alpha = nn.Parameter(torch.empty(n_emb, rank))
        self.w_beta = nn.Parameter(torch.empty(n_emb, rank))
        self.w_core = nn.Parameter(torch.empty(n_emb, n_vocab * rank * rank))

    def materialize(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def get_unnorm_prob(self, y: torch.Tensor) -> torch.Tensor:
        """Get the unnormalized probability of a sequence. (i.e, :math:`\tilde{p}(y)`)

        Args:
            y (torch.Tensor): Select vector. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Probability of the sequence. Shape: (batch_size,)
        """
        raise NotImplementedError

    def get_unnorm_prob_and_norm(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the unnormalized probability and normalization constant of a sequence. (i.e, :math:`\tilde{p}(y)` and :math:`Z`)

        Args:
            y (torch.Tensor): Select vector. Shape: (batch_size, seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probability of the sequence and normalization constant. Shape: (batch_size,) and (batch_size,)
        """
        raise NotImplementedError
