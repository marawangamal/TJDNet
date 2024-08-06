from typing import Tuple
import torch
import torch.nn as nn

from TJDNet.utils import umps_select_marginalize_batched, umps_materialize_batched


# MISC helpers


class MPSDist(nn.Module):
    def __init__(self, n_vocab: int, rank: int = 2, type="abs", init_method="unit_var"):
        super(MPSDist, self).__init__()
        assert type in ["born", "abs"]
        assert init_method in ["rand", "unit_var"]
        self.rank = rank
        self.init_method = init_method
        self.n_vocab = n_vocab
        self.n_born_machine_params = n_vocab * rank * rank + 2 * rank

        self.alpha = torch.randn(1, rank)
        self.beta = torch.randn(1, rank)
        self.core = nn.Parameter(torch.randn(1, rank, n_vocab, rank))

        # Cache
        self.norm_const = None

        if init_method == "unit_var":
            self._init_unit_var()

    def _init_unit_var(self):
        # Core
        core_data = torch.zeros_like(self.core, device=self.core.device)
        for i in range(self.n_vocab):
            core_data[0, :, i, :] = torch.eye(self.rank, device=self.core.device)

        # Alpha, Beta
        beta_data = (
            torch.randn_like(self.beta, device=self.beta.device)
            * 1
            / torch.sqrt(torch.tensor(self.rank))
        )

        alpha_data = (
            torch.randn_like(self.alpha, device=self.alpha.device)
            * 1
            / torch.sqrt(torch.tensor(self.rank))
        )

        self.alpha.data = alpha_data
        self.beta.data = beta_data
        self.core.data = core_data

    @staticmethod
    def _sample_one(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        max_len: int,
        batch_size: int = 1,
    ) -> torch.Tensor:
        selection_map = (
            torch.ones(batch_size, max_len, dtype=torch.long, device=alpha.device) * -1
        )  # (batch_size, max_len)
        for t in range(max_len):
            marginalize_mask = torch.cat(
                [
                    torch.zeros(
                        batch_size, t + 1, dtype=torch.long, device=alpha.device
                    ),
                    torch.ones(
                        batch_size,
                        max_len - t - 1,
                        dtype=torch.long,
                        device=alpha.device,
                    ),
                ],
                1,
            )
            # margins = 16 - 1 - t
            # selects = t
            p_vec_tilde = umps_select_marginalize_batched(
                alpha=alpha,
                beta=beta,
                core=core,
                selection_map=selection_map,
                marginalize_mask=marginalize_mask,
            )  # (batch_size, n_vocab)
            p_vec = p_vec_tilde / p_vec_tilde.sum(dim=-1, keepdim=True)
            indices = torch.multinomial(p_vec, 1)  # (batch_size, 1)
            selection_map[:, t] = indices.squeeze()

        # Get indices from selection_map
        return selection_map

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        func = {"born": lambda x: x**2, "abs": lambda x: torch.abs(x)}
        alpha = func["born"](self.alpha)
        beta = func["born"](self.beta)
        core = func["born"](self.core)
        return alpha, beta, core

    def sample(self, max_len: int, n_samples: int = 1) -> torch.Tensor:
        """Sample sequences from the MPS distribution.

        Args:
            max_len (int): Maximum length of the sequences.
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            torch.Tensor: Sampled sequences. Shape: (batch_size, max_len)
        """
        alpha, beta, core = self.get_params()
        samples = [
            self._sample_one(
                alpha=alpha,
                beta=beta,
                core=core,
                max_len=max_len,
            )
            for _ in range(n_samples)
        ]  # List of (1, max_len)
        return torch.stack(samples).squeeze(1)  # (n_samples, max_len)

    def get_unnorm_prob(self, y: torch.Tensor) -> torch.Tensor:
        """Get the unnormalized probability of a sequence. (i.e, :math:`\tilde{p}(y)`)

        Args:
            y (torch.Tensor): Select vector. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Probability of the sequence. Shape: (batch_size,)
        """
        alpha, beta, core = self.get_params()
        p_tilde = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
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
        alpha, beta, core = self.get_params()
        batch_size = y.shape[0]
        selection_map = torch.cat(
            [torch.ones(batch_size, 1, device=y.device, dtype=y.dtype) * -1, y[:, 1:]],
            1,
        )
        p_tilde_one = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            selection_map=selection_map,
            marginalize_mask=torch.zeros_like(y, device=y.device),
        )  # (batch_size, n_vocab)
        p_tilde = torch.stack([p_tilde_one[b, y[b, 0]] for b in range(batch_size)])
        marginalize_mask = torch.ones_like(y, device=y.device)
        marginalize_mask[:, 0] = 0
        z_one = umps_select_marginalize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            selection_map=torch.ones_like(y, device=y.device) * -1,
            marginalize_mask=marginalize_mask,
        )
        z = z_one.sum()
        self.norm_const = z

        assert torch.all(p_tilde >= 0), "p_tilde must be non-negative"
        assert torch.all(z >= 0), "Z must be non-negative"
        # assert torch.all(p_tilde <= z), "p_tilde must be less than Z"

        return p_tilde, z

    def materialize(
        self, normalize: bool = True, n_core_repititions: int = 3
    ) -> torch.Tensor:
        """Materialize the MPS distribution.

        Args:
            normalize (bool, optional): Whether to normalize the distribution. Defaults to True.

        Returns:
            torch.Tensor: Materialized distribution. Shape: (n_vocab,)
        """
        alpha, beta, core = self.get_params()
        return umps_materialize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            n_core_repititions=n_core_repititions,
            normalize=normalize,
        )