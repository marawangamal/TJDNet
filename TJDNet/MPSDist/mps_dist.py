from typing import Tuple
import torch
import torch.nn as nn

from TJDNet.utils import umps_select_marginalize_batched, umps_materialize_batched


# MISC helpers


class MPSDistBase:
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, core: torch.Tensor):
        """Initialize the MPS distribution.

        Args:
            alpha (torch.Tensor): Shape: (batch_size, rank)
            beta (torch.Tensor): Shape: (batch_size, rank)
            core (torch.Tensor): Shape: (batch_size, rank, n_vocab, rank)

        Note:
            If batch size mismatches runtime batch size, the parameters are repeated to match the batch size.
        """
        super().__init__()  # Initialize the base class
        self.alpha = alpha
        self.beta = beta
        self.core = core

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
            p_vec_tilde, _ = umps_select_marginalize_batched(
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

    def _make_batched_params(
        self,
        batch_size: int,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_batched, beta_batched, core_batched = (
            alpha,
            beta,
            core,
        )  # (batch_size, n_vocab)
        if alpha.shape[0] != batch_size:
            alpha_batched = alpha.repeat(batch_size, 1)
            beta_batched = beta.repeat(batch_size, 1)
            core_batched = core.repeat(batch_size, 1, 1, 1)
        return alpha_batched, beta_batched, core_batched

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

    # TODO: Add support scalar output from `umps_select_marginalize_batched`
    def get_unnorm_prob(
        self,
        y: torch.Tensor,
        apply_scale_factor: bool = True,
    ) -> torch.Tensor:
        """Get the unnormalized probability of a sequence. (i.e, :math:`\tilde{p}(y)`)

        Args:
            y (torch.Tensor): Select vector. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Probability of the sequence. Shape: (batch_size,)
        """
        alpha, beta, core = self.get_params()
        batch_size = y.shape[0]
        selection_map = torch.cat(
            [torch.ones(batch_size, 1, device=y.device, dtype=y.dtype) * -1, y[:, 1:]],
            1,
        )

        alpha_batched, beta_batched, core_batched = self._make_batched_params(
            batch_size, alpha, beta, core
        )

        p_tilde_one, z_list = umps_select_marginalize_batched(
            alpha=alpha_batched,
            beta=beta_batched,
            core=core_batched,
            selection_map=selection_map,
            marginalize_mask=torch.zeros_like(y, device=y.device),
            apply_scale_factor=apply_scale_factor,
        )  # (batch_size, n_vocab)
        p_tilde = torch.stack([p_tilde_one[b, y[b, 0]] for b in range(batch_size)])
        return p_tilde, z_list

    def get_norm_constant(
        self, y: torch.Tensor, apply_scale_factor: bool = True
    ) -> torch.Tensor:
        alpha, beta, core = self.get_params()
        marginalize_mask = torch.ones_like(y, device=y.device)
        marginalize_mask[:, 0] = 0
        batch_size = y.shape[0]

        alpha_batched, beta_batched, core_batched = self._make_batched_params(
            batch_size, alpha, beta, core
        )

        # Function umps_select_marginalize_batched needs to output a vector. So this marginalizes all except the first element.
        z_one, z_list = umps_select_marginalize_batched(
            alpha=alpha_batched,
            beta=beta_batched,
            core=core_batched,
            selection_map=torch.ones_like(y, device=y.device) * -1,
            marginalize_mask=marginalize_mask,
            apply_scale_factor=apply_scale_factor,
        )
        z = z_one.sum()
        return z, z_list

    def get_unnorm_prob_and_norm(
        self,
        y: torch.Tensor,
        apply_scale_factor: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the unnormalized probability and normalization constant of a sequence. (i.e, :math:`\tilde{p}(y)` and :math:`Z`)

        Args:
            y (torch.Tensor): Select vector. Shape: (batch_size, seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probability of the sequence and normalization constant. Shape: (batch_size,) and (batch_size,)
        """

        p_tilde, z_list_select = self.get_unnorm_prob(y, apply_scale_factor)
        z, z_list_norm = self.get_norm_constant(y, apply_scale_factor)

        assert torch.all(p_tilde >= 0), "p_tilde must be non-negative"
        assert torch.all(z >= 0), "Z must be non-negative"
        assert torch.all(p_tilde <= z), "p_tilde must be less than Z"
        return p_tilde, z, z_list_select, z_list_norm

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


class MPSDist(nn.Module, MPSDistBase):
    def __init__(
        self, n_vocab: int, rank: int = 2, positivity_func="abs", init_method="unit_var"
    ):
        super(MPSDist, self).__init__()
        assert positivity_func in ["square", "abs"]
        assert init_method in ["randn", "unit_var", "one_hot", "sparse"]
        self.rank = rank
        self.init_method = init_method
        self.n_vocab = n_vocab
        self.n_born_machine_params = n_vocab * rank * rank + 2 * rank

        alpha = torch.randn(1, rank)
        beta = torch.randn(1, rank)
        core = nn.Parameter(torch.randn(1, rank, n_vocab, rank))
        MPSDistBase.__init__(self, alpha, beta, core)

        if init_method == "unit_var":
            self._init_unit_var()
        elif init_method == "one_hot":
            self._init_one_hot()
        elif init_method == "sparse":
            self._init_one_sparse()

    def _init_unit_var(self):
        # Core
        core_data = torch.zeros_like(self.core, device=self.core.device)
        for i in range(self.n_vocab):
            # todo: add some noise
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

    def _init_one_hot(self, one_hot_idx: int = 0):
        # Core
        core_data = torch.zeros_like(self.core, device=self.core.device)
        core_data[0, :, 0, :] = torch.eye(self.rank, device=self.core.device)

        # Alpha, Beta
        beta_data = torch.zeros_like(self.beta, device=self.beta.device)
        beta_data[0, one_hot_idx] = 1

        alpha_data = torch.zeros_like(self.alpha, device=self.alpha.device)
        alpha_data[0, one_hot_idx] = 1

        self.alpha.data = alpha_data
        self.beta.data = beta_data
        self.core.data = core_data

    def _init_one_sparse(self):

        # want T(x1, ..., xn) = 1 with probability 0.5 and 0 otherwise

        # Core
        core_data = torch.zeros_like(self.core, device=self.core.device)
        for k in range(self.n_vocab):
            if k % 2 == 0:
                core_data[0, :, k, :] = torch.eye(self.rank, device=self.core.device)

        # Alpha, Beta
        beta_data = torch.zeros_like(self.beta, device=self.beta.device)
        alpha_data = torch.zeros_like(self.alpha, device=self.alpha.device)

        for k in range(self.rank):
            if k % 2 == 0:
                beta_data[0, k] = 1
                alpha_data[0, k] = 1

        self.alpha.data = alpha_data
        self.beta.data = beta_data
        self.core.data = core_data
