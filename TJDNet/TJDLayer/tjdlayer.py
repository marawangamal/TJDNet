from typing import Any, Callable, Optional, Tuple
import torch.nn as nn
import torch
from torch import Tensor

import logging

from .utils import create_core_ident, apply_id_transform

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs


def check_naninf(x: torch.Tensor, msg: Optional[str] = None, raise_error: bool = True):
    if torch.isnan(x).any() or torch.isinf(x).any():
        if raise_error:
            raise ValueError(f"NaN/Inf detected in tensor: {msg}")
        else:
            return True


class TTDist:
    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        n_core_repititions: int,
    ) -> None:
        """Tensor Train Parameterization of Joint Distribution.

        The TTDist class provides a way to represent and use a joint distribution using a tensor train parameterization.

        Args:
            alpha (torch.Tensor): Shape: (B, R).
            beta (torch.Tensor): Shape: (B, R).
            core (torch.Tensor): Shape: (B, R, D, R).
            n_core_repititions (int): Number of core repititions.
        """

        self.alpha = alpha
        self.beta = beta
        self.core = core
        self.batch_size = alpha.shape[0]
        self.n_core_repititions = n_core_repititions

    def _get_normalization_constant(self) -> torch.Tensor:
        """Get the normalization constant for the TTDist

        Returns:
            torch.Tensor: normalization constant. Shape (B,)
        """

        core_marginalized = torch.einsum(
            "bijk,j->bik",
            self.core,
            torch.ones(self.core.shape[2], device=self.core.device),
        )

        core_result = core_marginalized
        for i in range(self.n_core_repititions):
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
            self.alpha,
            core_result,
            self.beta,
        )

        # Assert: postive normalization constant
        assert torch.all(core_result > 0), "Normalization constant must be positive"

        return core_result

    def _select(self, indices: torch.Tensor) -> torch.Tensor:
        """Multi-index selection.

        Performs a multi-index selection on the TTDist object. Produces the unnormalized probability :math:`\tilde{P}(x_1, x_2, ..., x_n)` of the sequences corresponding to the indices.

        Args:
            indices (torch.Tensor): Shape: (B, n_core_repititions).

        Returns:
            torch.Tensor: Unnormalized probabilities of sequences corresponding to indices. Shape: (B,)
        """

        # Assert: indices are within the core size
        assert torch.all(
            indices < self.core.shape[2]
        ), "Indices must be within the core size"

        # Assert: indices are non-negative
        assert torch.all(indices >= 0), "Indices must be non-negative"

        batch_size = indices.size(0)
        cores_after_selection = [
            torch.stack([self.core[b, :, indices[b, i], :] for b in range(batch_size)])
            for i in range(self.n_core_repititions)
        ]  # [(B, R, R)] * n_core_repititions

        core_result = cores_after_selection[0]  # (B, R, R)
        for i in range(1, self.n_core_repititions):
            check_naninf(core_result, f"core_result_{i}")
            check_naninf(cores_after_selection[i], f"cores_after_selection_{i}")
            core_result = torch.einsum(
                "bik, bkj->bij",
                core_result,
                cores_after_selection[i],
            )

        core_result = torch.einsum(
            "bi, bij, bj->b",
            self.alpha,
            core_result,
            self.beta,
        )

        return core_result

    def _beam_search(self, n_beams: int = 5):

        if n_beams > self.core.shape[2]:
            logger.warning(
                f"Number of beams is greater than the core size. Setting beams to core size: {self.core.shape[2]}"
            )
            n_beams = self.core.shape[2]

        # Initialize beams
        beams: list[list[Tensor]] = [[torch.tensor(0)] for _ in range(n_beams)]
        beam_probs: list[Tensor] = [torch.tensor(1.0) for _ in range(n_beams)]

        # Assert only batch size of 1 is supported
        assert self.batch_size == 1, "Batch size must be 1 for beam search"

        beta = self.beta[0]
        alpha = self.alpha[0]
        core = self.core[0]
        core_marginalized = torch.einsum(
            "ijk,j->ik", core, torch.ones(core.shape[1], device=core.device)
        )

        for i_pos in range(self.n_core_repititions):
            for _ in range(n_beams):
                curr_beam = beams.pop(0)
                beam_probs.pop(0)

                # 1. Get the selection tensor
                selection_tensor = alpha.reshape(1, -1)  # [1, R]
                for i_selection in range(i_pos):
                    # [1, R] @ [R, R] = [1, R]
                    selection_tensor = (
                        selection_tensor @ core[:, curr_beam[i_selection], :]
                    )

                # 2. Get the middle tensor
                current_tensor = core  # [R, D, R]

                # 3. Get the marginalized tensor
                marginalized_tensor = beta.reshape(-1, 1)  # [R, 1]
                for _ in range(i_pos + 1, self.n_core_repititions):
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

    def get_prob_and_norm(self, indices) -> Tuple[Tensor, Tensor]:
        unormalized_probs = self._select(indices)
        check_naninf(unormalized_probs, f"get_prob:unormalized_probs")

        normalization_constant = self._get_normalization_constant()
        check_naninf(normalization_constant, f"get_prob:normalization_constant")

        # max_unorm_prob = torch.max(torch.abs(unormalized_probs)).item()
        # min_unorm_prob = torch.min(torch.abs(unormalized_probs)).item()
        # logger.debug(f"Max unnormalized prob: {max_unorm_prob:.4f}")
        # logger.debug(f"Min unnormalized prob: {min_unorm_prob:.4f}")
        # logger.debug(f"Normalization constant: {normalization_constant}")

        return unormalized_probs, normalization_constant

    def sample(self):
        return torch.randint(
            0, self.core.shape[2], (self.batch_size, self.n_core_repititions)
        )  # [B, n_core_repititions]

    def beam_search(self, n_beams: int = 5) -> torch.Tensor:
        """Beam search using the TTDist joint distribution.

        Args:
            n_beams (int, optional): Number of beams. Defaults to 5.

        Returns:
            torch.Tensor: Beam search results. Shape: (B, n_core_repititions)
        """
        beams, _ = self._beam_search(n_beams=n_beams)
        return torch.tensor(beams)


class TJDLayer(nn.Module):
    def __init__(
        self,
        emb_size,
        rank: int = 2,
        vocab_size: int = 128,
        identity_transform_ids: list[int] = [50256],
        identity_transform_label_id_token_id: dict[int, int] = {-100: 50256},
        *args,
        **kwargs,
    ):
        """Tensor Train Joint Distribution Layer

        Args:
            emb_size (_type_): Embedding size.
            rank (int, optional): Rank of the TT decomposition. Defaults to 2.
            vocab_size (int, optional): Vocabulary size. Defaults to 128.
        """
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.rank = rank
        self.vocab_size = vocab_size
        self.w_alpha = nn.Parameter(torch.randn(emb_size, rank))
        self.w_beta = nn.Parameter(torch.randn(emb_size, rank))
        self.w_vocab = nn.Parameter(torch.randn(emb_size, vocab_size * rank * rank))
        self.identity_transform_ids = identity_transform_ids
        self.identity_transform_label_id_token_id = identity_transform_label_id_token_id

    def _compute_loss(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute loss for TT JD layer

        Args:
            alpha (torch.Tensor): Shape: (B, R).
            beta (torch.Tensor): Shape: (B, R).
            core (torch.Tensor): Shape: (B, R, D, R).
            target (torch.Tensor): Shape: (B, T).
            reduction (str, optional): Reduction type. Defaults to "mean".

        Returns:
            torch.Tensor: Loss. Shape: (B,)
        """

        # Transform target using self.identity_transform_label_id_token_id
        if self.identity_transform_label_id_token_id is not None:
            # target [B, T] tensor
            # elementwise transform target[k] = self.identity_transform_label_id_token_id[target[k]] if target[k] in self.identity_transform_label_id_token_id
            target = apply_id_transform(
                target, self.identity_transform_label_id_token_id
            )

        output_size = target.size(1)
        ttdist = TTDist(alpha, beta, core, output_size)
        prob_tilde, norm_constant = ttdist.get_prob_and_norm(target)
        log_prob = torch.log(prob_tilde) - torch.log(norm_constant)
        loss = -log_prob
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _get_preds(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        ttdist = TTDist(alpha, beta, core, output_size)
        return ttdist.sample()  # (B, output_size)

    def _get_tt_params(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, seq_len, _ = x.shape
        alpha = (
            (x.reshape(-1, x.shape[-1]) @ self.w_alpha)
            .reshape(batch_size, seq_len, self.rank)
            .mean(1)
        )
        beta = (
            (x.reshape(-1, x.shape[-1]) @ self.w_beta)
            .reshape(batch_size, seq_len, self.rank)
            .mean(1)
        )
        dcore = (
            (x.reshape(-1, x.shape[-1]) @ self.w_vocab)
            .reshape(batch_size, seq_len, self.rank, self.vocab_size, self.rank)
            .mean(1)
        )  # (B, R, D, R)

        alpha = torch.relu(alpha) + 1e-3
        beta = torch.relu(beta) + 1e-3
        dcore = torch.relu(dcore) * (1 / seq_len) + 1e-3

        # Alter core s.t core[b, :, d, :] = I for d in identity_transform_ids
        core_ident = create_core_ident(
            batch_size=batch_size, vocab_size=self.vocab_size, rank=self.rank
        ).to(dcore.device)
        mask = torch.ones_like(core_ident).to(dcore.device)
        mask[:, :, self.identity_transform_ids, :] = 0
        core = core_ident + mask * dcore

        alpha_min, alpha_max = torch.min(torch.abs(alpha)), torch.max(torch.abs(alpha))
        beta_min, beta_max = torch.min(torch.abs(beta)), torch.max(torch.abs(beta))
        core_min, core_max = torch.min(torch.abs(core)), torch.max(torch.abs(core))

        logger.debug(f"Alpha min: {alpha_min:.4f}, Alpha max: {alpha_max:.4f}")
        logger.debug(f"Beta min: {beta_min:.4f}, Beta max: {beta_max:.4f}")
        logger.debug(f"Core min: {core_min:.4f}, Core max: {core_max:.4f}")

        return alpha, beta, core

    def get_preds(
        self,
        input_embs: torch.Tensor,
        max_length: int = 100,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Get predictions from TT JD layer

        Args:
            input_embs (torch.Tensor): Input embeddings. Shape: (B, T, D)
            max_length (int, optional): Maximum length of the output. Defaults to 100.

        Returns:
            torch.Tensor: Prediction indices. Shape: (B, T)
        """
        x = input_embs
        alpha, beta, core = self._get_tt_params(x)
        for tensor, tensor_name in zip([alpha, beta, core], ["alpha", "beta", "core"]):
            check_naninf(tensor, f"forward:{tensor_name}")
        preds = self._get_preds(alpha, beta, core, max_length)
        return preds

    def forward(
        self,
        input_embs: torch.Tensor,
        label_ids: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for TT JD layer

        Args:
            input_embs (torch.Tensor): Input embeddings. Shape: (B, T, D)
            label_ids (torch.Tensor): Label ids. Shape: (B, T)

        Returns:
            torch.Tensor: Loss. Shape: (B,)
        """

        x = input_embs
        y = label_ids

        alpha, beta, core = self._get_tt_params(x)
        for tensor, tensor_name in zip([alpha, beta, core], ["alpha", "beta", "core"]):
            check_naninf(tensor, f"forward:{tensor_name}")

        loss = self._compute_loss(alpha, beta, core, y)
        check_naninf(loss, f"forward:loss")
        return loss
