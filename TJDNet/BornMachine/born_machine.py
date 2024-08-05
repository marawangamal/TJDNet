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

        self.alpha = nn.Parameter(torch.randn(1, rank))
        self.beta = nn.Parameter(torch.randn(1, rank))
        self.core = nn.Parameter(torch.randn(1, rank, n_vocab, rank))

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
        p_tilde = umps_select_marginalize_batched(
            alpha=self.alpha,
            beta=self.beta,
            core=self.core,
            selection_map=y,
            marginalize_mask=torch.zeros_like(y, device=y.device),
        )
        return p_tilde

    def get_unnorm_prob_and_norm(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the unnormalized probability and normalization constant of a sequence. (i.e, :math:`\tilde{p}(y)` and :math:`Z`)

        Args:
            y (torch.Tensor): Select vector. Shape: (batch_size, seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probability of the sequence and normalization constant. Shape: (batch_size,) and (batch_size,)
        """
        p_tilde = umps_select_marginalize_batched(
            alpha=self.alpha,
            beta=self.beta,
            core=self.core,
            selection_map=y,
            marginalize_mask=torch.zeros_like(y, device=y.device),
        )
        marginalize_mask = torch.ones_like(y, device=y.device)
        marginalize_mask[:, 0] = 0
        z_one = umps_select_marginalize_batched(
            alpha=self.alpha,
            beta=self.beta,
            core=self.core,
            selection_map=torch.ones_like(y, device=y.device) * -1,
            marginalize_mask=marginalize_mask,
        )
        z = z_one.sum()
        return p_tilde, z
