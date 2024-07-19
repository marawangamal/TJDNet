from typing import Optional, Dict
import logging


from .utils import (
    check_naninf,
    select_and_marginalize_uMPS,
)


import torch
from torch import Tensor


from typing import List, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TTDist:
    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        n_core_repititions: int,
        repeat_batch_size: Optional[int] = None,
        eps: float = 1e-10,
        norm_method: str = "relu",
        norm_method_alpha: str = "relu",
    ) -> None:
        """Tensor Train Parameterization of Joint Distribution.

        The TTDist class provides a way to represent and use a joint distribution using a tensor train parameterization.

        Args:
            alpha (torch.Tensor): Shape: (B, R).
            beta (torch.Tensor): Shape: (B, R).
            core (torch.Tensor): Shape: (B, R, D, R).
            n_core_repititions (int): Number of core repititions.
        """

        check_naninf(alpha, f"TTDist:alpha")
        check_naninf(beta, f"TTDist:beta")
        check_naninf(core, f"TTDist:core")

        assert norm_method in [
            "relu",
            "abs",
            "softmax",
            "sigmoid",
        ], f"Normalization method must be one of {norm_method}"
        self.norm_method = norm_method
        self.norm_method_alpha = norm_method_alpha

        self.precond_func = {
            "relu": torch.relu,
            "abs": torch.abs,
            "sigmoid": torch.sigmoid,
            "softmax": torch.exp,
        }[self.norm_method]

        self.precond_func_alpha = {
            "relu": torch.relu,
            "abs": torch.abs,
            "sigmoid": torch.sigmoid,
            "softmax": torch.exp,
        }[self.norm_method_alpha]

        self.alpha = self.precond_func_alpha(alpha) + eps
        self.beta = self.precond_func_alpha(beta) + eps
        self.core = self.precond_func(core) + eps
        self.batch_size = alpha.shape[0]
        self.n_core_repititions = n_core_repititions

        if repeat_batch_size:
            self.alpha = self.alpha.repeat(repeat_batch_size, 1)
            self.beta = self.beta.repeat(repeat_batch_size, 1)
            self.core = self.core.repeat(repeat_batch_size, 1, 1, 1)

    @staticmethod
    def _get_normalization_constant(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        n_core_repititions: int,
    ) -> torch.Tensor:
        """Get the normalization constant for the TTDist

        Returns:
            torch.Tensor: normalization constant. Shape (B,)
        """

        core_marginalized = torch.einsum(
            "bijk,bj->bik",
            core,
            torch.ones(core.shape[0], core.shape[2], device=core.device),
        )

        core_result = core_marginalized
        for i in range(max(n_core_repititions - 1, 0)):
            check_naninf(core_result, f"_get_normalization_constant:core_result_{i}")
            check_naninf(
                core_marginalized,
                f"_get_normalization_constant:core_marginalized{i}",
            )
            core_result = torch.einsum(
                "bik, bkj->bij",
                core_result,
                core_marginalized,
            )

        core_result = torch.einsum(
            "bi, bij, bj->b",
            alpha,
            core_result,
            beta,
        )

        return core_result

    @staticmethod
    def _select(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        n_core_repititions: int,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-index selection.

        Performs a multi-index selection on the TTDist object. Produces the unnormalized probability :math:`\tilde{P}(x_1, x_2, ..., x_n)` of the sequences
        corresponding to the indices.

        Args:
            indices (torch.Tensor): Shape: (B, n_core_repititions).

        Returns:
            torch.Tensor: Unnormalized probabilities of sequences corresponding to indices. Shape: (B,)
        """

        batch_size = indices.size(0)
        cores_after_selection = [
            torch.stack([core[b, :, indices[b, i], :] for b in range(batch_size)])
            for i in range(n_core_repititions)
        ]  # [(B, R, R)] * n_core_repititions

        core_result = cores_after_selection[0]  # (B, R, R)
        for i in range(1, n_core_repititions):
            check_naninf(core_result, f"core_result_{i}")
            check_naninf(cores_after_selection[i], f"cores_after_selection_{i}")
            core_result = torch.einsum(
                "bik, bkj->bij",
                core_result,
                cores_after_selection[i],
            )

        core_result = torch.einsum(
            "bi, bij, bj->b",
            alpha,
            core_result,
            beta,
        )

        return core_result

    @staticmethod
    def _beam_search(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        n_beams: int = 5,
        n_core_repititions: int = 5,
    ):

        if n_beams > core.shape[2]:
            logger.warning(
                f"Number of beams is greater than the core size. Setting beams to core size: {core.shape[2]}"
            )
            n_beams = core.shape[2]

        # Initialize beams
        beams: List[List[Tensor]] = [[torch.tensor(0)] for _ in range(n_beams)]
        beam_probs: List[Tensor] = [torch.tensor(1.0) for _ in range(n_beams)]

        # Assert only batch size of 1 is supported
        # assert self.batch_size == 1, "Batch size must be 1 for beam search"
        assert alpha.shape[0] == 1, "Batch size must be 1 for beam search"

        _beta = beta[0]
        _alpha = alpha[0]
        _core = core[0]
        core_marginalized = torch.einsum(
            "ijk,j->ik", _core, torch.ones(_core.shape[1], device=_core.device)
        )

        for i_pos in range(n_core_repititions):
            for _ in range(n_beams):
                curr_beam = beams.pop(0)
                beam_probs.pop(0)

                # 1. Get the selection tensor
                selection_tensor = _alpha.reshape(1, -1)  # [1, R]
                for i_selection in range(i_pos):
                    # [1, R] @ [R, R] = [1, R]
                    selection_tensor = (
                        selection_tensor @ _core[:, curr_beam[i_selection], :]
                    )

                # 2. Get the middle tensor
                current_tensor = _core  # [R, D, R]

                # 3. Get the marginalized tensor
                marginalized_tensor = _beta.reshape(-1, 1)  # [R, 1]
                for _ in range(i_pos + 1, n_core_repititions):
                    # [R, R] @ [R, 1] = [R, 1]
                    marginalized_tensor = core_marginalized @ marginalized_tensor

                # 4. Perform the total contraction
                #  [1, R] @ [R, D, R] @ [R, 1] = [1, D, 1]
                total_tensor = torch.einsum(
                    "r,rdR,R->d",
                    selection_tensor.squeeze(),
                    current_tensor,
                    marginalized_tensor.squeeze(),
                )

                argtop_n_beams = torch.argsort(total_tensor, descending=True)[
                    :n_beams
                ]  # [n_beams]
                valstop_n_beams = total_tensor[argtop_n_beams]  # [n_beams]

                new_beams = [
                    curr_beam + [argtop_n_beams[ii_beam]] for ii_beam in range(n_beams)
                ]
                new_beam_probs = [
                    valstop_n_beams[ii_beam] for ii_beam in range(n_beams)
                ]

                # Delete current beam and replace with new beams
                beams += new_beams
                beam_probs += new_beam_probs

            # Keep only the top n_beams
            top_n_beams = torch.argsort(torch.tensor(beam_probs), descending=True)[
                :n_beams
            ]
            beams = [beams[i_top_beam] for i_top_beam in top_n_beams]
            beam_probs = [beam_probs[i_top_beam] for i_top_beam in top_n_beams]

        # Remove the first element of each beam
        for i_beam in range(n_beams):
            beams[i_beam] = beams[i_beam][1:]
        return beams, beam_probs

    def get_prob_and_norm(self, indices: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Get the unnormalized probability and normalization constant of the TTDist.

        Args:
            indices (torch.Tensor): Indices of the sequences. Shape: (B, n_core_repititions).

        Returns:
            Tuple[Tensor, Tensor]: Unnormalized probabilities and normalization constant. Shapes: (B,), (B,)
        """

        # Assert: indices are within the core size
        assert torch.all(
            indices < self.core.shape[2]
        ), "Indices must be within the core size"

        # Assert: indices are non-negative
        assert torch.all(indices >= 0), "Indices must be non-negative"
        check_naninf(self.alpha, f"get_prob_and_norm:alpha")
        check_naninf(self.beta, f"get_prob_and_norm:beta")
        check_naninf(self.core, f"get_prob_and_norm:core")

        unormalized_probs = self._select(
            alpha=self.alpha,
            beta=self.beta,
            core=self.core,
            n_core_repititions=self.n_core_repititions,
            indices=indices,
        )
        check_naninf(unormalized_probs, f"get_prob:unormalized_probs")

        normalization_constant = self._get_normalization_constant(
            alpha=self.alpha,
            beta=self.beta,
            core=self.core,
            n_core_repititions=self.n_core_repititions,
        )
        check_naninf(normalization_constant, f"get_prob:normalization_constant")

        # max_unorm_prob = torch.max(torch.abs(unormalized_probs)).item()
        # min_unorm_prob = torch.min(torch.abs(unormalized_probs)).item()
        # logger.debug(f"Max unnormalized prob: {max_unorm_prob:.4f}")
        # logger.debug(f"Min unnormalized prob: {min_unorm_prob:.4f}")
        # logger.debug(f"Normalization constant: {normalization_constant}")

        return unormalized_probs, normalization_constant

    def sample(self, n_samples: int = 1, batch_idx: int = 0) -> torch.Tensor:
        selection_ids = {}
        for i in range(self.n_core_repititions):
            marginalize_ids = list(range(i + 1, self.n_core_repititions))
            p_vec_tilde = select_and_marginalize_uMPS(
                self.alpha[batch_idx],
                self.beta[batch_idx],
                self.core[batch_idx],
                self.n_core_repititions,
                selection_ids=selection_ids,
                marginalize_ids=marginalize_ids,
            )
            if p_vec_tilde is None:
                raise ValueError("Invalid selection and marginalization indices")
            p_vec = p_vec_tilde / p_vec_tilde.sum()
            idx = torch.multinomial(p_vec, 1)
            selection_ids[i] = idx.item()
        return torch.tensor([selection_ids[i] for i in range(self.n_core_repititions)])

    def materialize(self):
        """Materialize the TTDist object.

        Returns:
            torch.Tensor: Materialized joint distribution. Shape: (B, D, D, ..., D)
        """

        rank_size = self.core.shape[1]
        batch_size = self.core.shape[0]
        vocab_size = self.core.shape[2]

        alpha, beta, core = self.alpha, self.beta, self.core
        # Test
        resultFirst = torch.einsum("i, idj, j->d", alpha[0], core[0], beta[0])

        result = torch.einsum(
            "bi, bidj->bdj",
            alpha,
            core,
        )

        for i in range(1, self.n_core_repititions):
            result = torch.einsum(
                "bdi, bivj->bdvj",
                result,
                core,
            )
            result = result.reshape(batch_size, -1, rank_size)

        result = torch.einsum(
            "bdj, bj->bd",
            result,
            beta,
        )

        # Break out all vocab_size dimensions
        result = result.reshape(
            batch_size, *[vocab_size for _ in range(self.n_core_repititions)]
        )
        return result

    def beam_search(self, n_beams: int = 5) -> torch.Tensor:
        """Beam search using the TTDist joint distribution.

        Args:
            n_beams (int, optional): Number of beams. Defaults to 5.

        Returns:
            torch.Tensor: Beam search results. Shape: (B, n_core_repititions)
        """
        beams, _ = self._beam_search(
            self.alpha,
            self.beta,
            self.core,
            n_beams=n_beams,
            n_core_repititions=self.n_core_repititions,
        )
        return torch.tensor(beams)
