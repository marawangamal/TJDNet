from typing import Tuple
import torch
import torch.nn as nn

from TJDNet.utils import umps_select_marginalize_batched, umps_materialize_batched


class MPSDist(nn.Module):
    def __init__(self, n_emb: int, n_vocab: int, rank: int = 2, *args, **kwargs):
        super(MPSDist, self).__init__()
        self.n_emb = n_emb
        self.n_vocab = n_vocab
        self.n_born_machine_params = n_vocab * rank * rank + 2 * rank

        self.alpha = nn.Parameter(torch.randn(1, rank))
        self.beta = nn.Parameter(torch.randn(1, rank))
        self.core = nn.Parameter(torch.randn(1, rank, n_vocab, rank))

    def _sample_one(self, max_len: int, batch_size: int = 1) -> torch.Tensor:
        selection_map = (
            torch.ones(batch_size, max_len, dtype=torch.long, device=self.alpha.device)
            * -1
        )  # (batch_size, max_len)
        for t in range(max_len):
            marginalize_mask = torch.concatenate(
                [
                    torch.zeros(
                        batch_size, t, dtype=torch.long, device=self.alpha.device
                    ),
                    torch.ones(
                        batch_size,
                        max_len - t,
                        dtype=torch.long,
                        device=self.alpha.device,
                    ),
                ]
            )
            p_vec_tilde = umps_select_marginalize_batched(
                alpha=self.alpha,
                beta=self.beta,
                core=self.core,
                selection_map=selection_map,
                marginalize_mask=marginalize_mask,
            )  # (batch_size, n_vocab)
            p_vec = p_vec_tilde / p_vec_tilde.sum(dim=-1, keepdim=True)
            indices = torch.multinomial(p_vec, 1)  # (batch_size, 1)
            selection_map[:, t] = indices.squeeze()

        # Get indices from selection_map
        return selection_map

    def sample(self, max_len: int, n_samples: int = 1) -> torch.Tensor:
        """Sample sequences from the MPS distribution.

        Args:
            max_len (int): Maximum length of the sequences.
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            torch.Tensor: Sampled sequences. Shape: (batch_size, max_len)
        """
        samples = [
            self._sample_one(max_len) for _ in range(n_samples)
        ]  # List of (1, max_len)
        return torch.stack(samples).squeeze(1)  # (n_samples, max_len)

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

    def materialize(self, normalize: bool = True) -> torch.Tensor:
        return umps_materialize_batched(
            alpha=self.alpha,
            beta=self.beta,
            core=self.core,
            n_core_repititions=self.n_vocab,
            normalize=normalize,
        )
