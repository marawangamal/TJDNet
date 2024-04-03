from typing import Any, Callable, Optional, Tuple
import torch.nn as nn
import torch
from torch import Tensor

import logging

logger = logging.getLogger(__name__)


def normalize_matrix(matrix: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # Make matrix positive and all columns sum to 1
    matrix = torch.abs(matrix)
    return matrix / matrix.sum(dim=dim, keepdim=True)


class BTTN:
    def __init__(self, alpha, beta, core, n_core_repititions):
        """Batch Tensor Train Network

        Args:
            alpha: [B, R]
            beta: [B, R]
            core: [B, R, D, R]

        """
        self.alpha = alpha
        self.beta = beta
        self.core = core
        self.batch_size = alpha.shape[0]
        self.n_core_repititions = n_core_repititions

    def select(self, indices):
        """Select elements from BTTN

        Args:
            indices: [B, n_core_repititions]

        Returns:
            results: [B,] // probabilities of sequences corresponding to indices

        """
        # Ensure that indices are in the correct range
        assert torch.all(
            indices < self.core.shape[2]
        ), "Indices must be less than the core size"
        assert torch.all(
            indices[indices != -100] >= 0
        ), "Indices must be greater than or equal to 0"
        results = []
        for b in range(self.batch_size):
            cores = [
                self.core[b, :, indices[b, k], :]
                for k in range(self.n_core_repititions)
            ]
            # matrix multiplication all cores together
            result = normalize_matrix(cores[0]) @ normalize_matrix(
                self.alpha[b].reshape(-1, 1)
            )  # [r, 1]
            for core in cores[1:]:
                result = normalize_matrix(core) @ result  # [r, r] @ [r, 1] = [r, 1]
            result = normalize_matrix(self.beta[b].reshape(-1, 1)).T @ result  # [r, 1]
            results.append(result.squeeze())

        return torch.stack(results)  # [B,]

    def argmax(self):
        return torch.randint(
            0, self.core.shape[2], (self.batch_size, self.n_core_repititions)
        )  # [B, n_core_repititions]

    def beam_search(self, n_beams: int = 5):

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
        core_marginalized = torch.einsum("ijk,j->ik", core, torch.ones(core.shape[1]))

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


class TJDLayer(nn.Module):
    def __init__(self, emb_size, rank: int = 2, vocab_size: int = 128, *args, **kwargs):
        """Tensor Train Joint Distribution Layer"""
        # Define TT JD parameters
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.rank = rank
        self.vocab_size = vocab_size
        self.w_alpha = nn.Parameter(torch.randn(emb_size, rank))
        self.w_beta = nn.Parameter(torch.randn(emb_size, rank))
        self.w_vocab = nn.Parameter(torch.randn(emb_size, vocab_size * rank * rank))

    def _compute_loss(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        output_size: int,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Get probabilities from Tensor Train representation of Joint Distribution

        Args:
            alpha: [B, R]
            beta: [B, R]
            core: [B, R, D, R] // vocab_size
            target: [B, output_size]

        Returns:
            probs: [output_size * vocab_size]
        """
        bttn = BTTN(alpha, beta, core, output_size)
        probs = bttn.select(target)
        loss = -torch.log(probs)
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
        """Get probabilities from Tensor Train representation of Joint Distribution

        Args:
            alpha: [B, R]
            beta: [B, R]
            core: [B, R, D, R] // vocab_size

        Returns:
            probs: [output_size * vocab_size]
        """
        btn = BTTN(alpha, beta, core, output_size)
        return btn.argmax()  # [B, output_size]

    def get_preds(
        self,
        input_embs: torch.Tensor,
        max_length: int = 100,
        *args,
        **kwargs,
    ):
        """Forward pass for TT JD layer

        Args:
            x: [B, T, D]

        Returns:
            preds: [B, output_size]

        """
        x = input_embs
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
        core = (
            (x.reshape(-1, x.shape[-1]) @ self.w_vocab)
            .reshape(batch_size, seq_len, self.rank, self.vocab_size, self.rank)
            .mean(1)
        )
        preds = self._get_preds(alpha, beta, core, seq_len)
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
            x: torch.Tensor([B, T, d])

        Returns:
            loss: torch.Tensor([1,])

        """
        x = input_embs
        y = label_ids
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
        core = (
            (x.reshape(-1, x.shape[-1]) @ self.w_vocab)
            .reshape(batch_size, seq_len, self.rank, self.vocab_size, self.rank)
            .mean(1)
        )
        loss = self._compute_loss(alpha, beta, core, seq_len, y)
        return loss
