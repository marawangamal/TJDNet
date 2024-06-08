from typing import Any, List, Optional, Tuple, Dict, Union
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import init

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
        repeat_batch_size: Optional[int] = None,
        eps: float = 1e-10,
        norm_method: str = "relu",
    ) -> None:
        """Tensor Train Parameterization of Joint Distribution.

        The TTDist class provides a way to represent and use a joint distribution using a tensor train parameterization.

        Args:
            alpha (torch.Tensor): Shape: (B, R).
            beta (torch.Tensor): Shape: (B, R).
            core (torch.Tensor): Shape: (B, R, D, R).
            n_core_repititions (int): Number of core repititions.
        """

        assert norm_method in [
            "relu",
            "abs",
            "softmax",
            "sigmoid",
        ], f"Normalization method must be one of {norm_method}"
        self.norm_method = norm_method

        self.precond_func = {
            "relu": torch.relu,
            "abs": torch.abs,
            "sigmoid": torch.sigmoid,
            "softmax": lambda x: x,
        }[self.norm_method]

        self.alpha = self.precond_func(alpha) + eps
        self.beta = self.precond_func(beta) + eps
        self.core = self.precond_func(core) + eps
        self.batch_size = alpha.shape[0]
        self.n_core_repititions = n_core_repititions

        if repeat_batch_size:
            self.alpha = self.alpha.repeat(repeat_batch_size, 1)
            self.beta = self.beta.repeat(repeat_batch_size, 1)
            self.core = self.core.repeat(repeat_batch_size, 1, 1, 1)

    def _get_normalization_constant(self) -> torch.Tensor:
        """Get the normalization constant for the TTDist

        Returns:
            torch.Tensor: normalization constant. Shape (B,)
        """

        core_marginalized = torch.einsum(
            "bijk,bj->bik",
            self.core,
            torch.ones(self.core.shape[0], self.core.shape[2], device=self.core.device),
        )

        core_result = core_marginalized
        for i in range(max(self.n_core_repititions - 1, 0)):
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

        # Get mask where core_result is zero and set to e-10
        zero_mask = core_result == 0
        if zero_mask.any():
            logger.warning(f"Normalization constant has zeros.")

        return core_result

    def _select(self, indices: torch.Tensor) -> torch.Tensor:
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
        beams: List[List[Tensor]] = [[torch.tensor(0)] for _ in range(n_beams)]
        beam_probs: List[Tensor] = [torch.tensor(1.0) for _ in range(n_beams)]

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

    def get_softmax_prob(self, indices) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def sample(self):
        return torch.randint(
            0, self.core.shape[2], (self.batch_size, self.n_core_repititions)
        )  # [B, n_core_repititions]

    def materialize(self):
        """Materialize the TTDist object.

        Returns:
            torch.Tensor: Materialized joint distribution. Shape: (B, D, D, ..., D)
        """

        rank_size = self.core.shape[1]
        batch_size = self.core.shape[0]
        vocab_size = self.core.shape[2]

        result = torch.einsum(
            "bi, bidj->bdj",
            self.alpha,
            self.core,
        )

        for i in range(1, self.n_core_repititions):
            result = torch.einsum(
                "bdi, bivj->bdvj",
                result,
                self.core,
            )
            result = result.reshape(batch_size, -1, rank_size)

        result = torch.einsum(
            "bdj, bj->bd",
            result,
            self.beta,
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
        beams, _ = self._beam_search(n_beams=n_beams)
        return torch.tensor(beams)


class BasicTJDLayerOutput:
    def __init__(self, loss: torch.Tensor, prob: torch.Tensor) -> None:
        self.loss = loss
        self.prob = prob


class BasicTJDLayer(nn.Module):
    def __init__(self, rank: int, vocab_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.alpha = nn.Parameter(torch.abs(torch.randn(rank)))
        self.beta = nn.Parameter(torch.abs(torch.randn(rank)))
        self.core = nn.Parameter(torch.abs(torch.randn(rank, vocab_size, rank)))
        self.eps = 1e-10

    def get_prob(self, target: torch.Tensor) -> torch.Tensor:
        ttdist = TTDist(self.alpha, self.beta, self.core, 1)
        prob_tilde, norm_constant = ttdist.get_prob_and_norm(target)
        return prob_tilde / norm_constant

    def beam_argmax(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        core: torch.Tensor,
        n_repetitions: int,
    ):
        """Computes argmax_x P(x1, x2, ... xN).

        Probability is parameterized with a TT distribution.

        Args:
            alpha (torch.Tensor): Shape: (B, R).
            beta (torch.Tensor): Shape: (B, R).
            core (torch.Tensor): Shape: (B, R, V, R).
        """

        ttdist = TTDist(alpha, beta, core, n_repetitions)
        max_seqs = ttdist.beam_search(n_beams=1)
        return max_seqs

    def forward(self, label_ids: torch.Tensor) -> BasicTJDLayerOutput:
        batch_size, n_repetitions = label_ids.shape
        batched_alpha = self.alpha.repeat(batch_size, 1)
        batched_beta = self.beta.repeat(batch_size, 1)
        batched_core = self.core.repeat(batch_size, 1, 1, 1)
        ttdist = TTDist(batched_alpha, batched_beta, batched_core, n_repetitions)
        prob_tilde, norm_constant = ttdist.get_prob_and_norm(label_ids)
        loss = -torch.log(prob_tilde + self.eps) + torch.log(norm_constant + self.eps)
        return BasicTJDLayerOutput(loss.mean(), prob_tilde / norm_constant)


class TJDLayer(nn.Module):
    def __init__(
        self,
        emb_size,
        rank: int = 2,
        vocab_size: int = 128,
        identity_transform_ids: List[int] = [50256],
        identity_transform_label_id_token_id: Dict[int, int] = {-100: 50256},
        mode: str = "tjd",  #  ["tjd", "lm", "tjd-lm", "tjd-lm-plus"]
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
        self._validate_init_args(
            emb_size=emb_size,
            rank=rank,
            vocab_size=vocab_size,
            identity_transform_ids=identity_transform_ids,
            identity_transform_label_id_token_id=identity_transform_label_id_token_id,
            mode=mode,
            *args,
            **kwargs,
        )
        self.emb_size = emb_size
        self.rank = rank
        self.vocab_size = vocab_size
        self.identity_transform_ids = identity_transform_ids
        self.identity_transform_label_id_token_id = identity_transform_label_id_token_id
        self.mode = mode

        self.w_alpha = nn.Parameter(torch.empty(emb_size, rank))
        self.w_beta = nn.Parameter(torch.empty(emb_size, rank))
        self.w_vocab = nn.Parameter(torch.empty(emb_size, vocab_size * rank * rank))

        # Initialize with Kaiming Normal for ReLU activations
        init.kaiming_normal_(self.w_alpha, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.w_beta, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.w_vocab, mode="fan_out", nonlinearity="relu")

    def _validate_init_args(
        self,
        emb_size,
        rank: int = 2,
        vocab_size: int = 128,
        identity_transform_ids: List[int] = [50256],
        identity_transform_label_id_token_id: Dict[int, int] = {-100: 50256},
        mode: str = "tjd",  # ["tjd", "lm"]
        *args,
        **kwargs,
    ):
        assert emb_size > 0, "emb_size must be positive"
        assert rank > 0, "rank must be positive"
        assert vocab_size > 0, "vocab_size must be positive"
        assert mode in [
            "tjd",
            "tjd-ident",
            "tjd-bounded",
            "ce",
            "ce-plus",
            "log-softmax",
            "log-prob",
        ], "mode must be either 'ce', 'log-softmax', or 'log-prob'"

    def _validate_forward_args(self, input_embs: torch.Tensor, label_ids: torch.Tensor):
        # Assert: input_embs and label_ids have the same batch size
        assert input_embs.size(0) == label_ids.size(
            0
        ), "Batch size mismatch between input_embs and label_ids"

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

        if self.mode == "tjd-bounded":
            # Take another random target and compute triplet loss
            target_neg = torch.randint_like(
                target, 0, self.vocab_size, device=target.device
            )
            prob_tilde_neg, norm_constant_neg = ttdist.get_prob_and_norm(target_neg)
            prob_tilde_neg_bounded = torch.clamp(prob_tilde_neg, 0, 1)
            prob_tilde_bounded = torch.clamp(prob_tilde, 0, 1)

            # loss_inv = prob_tilde_bounded + torch.abs(
            #     prob_tilde_bounded - prob_tilde_neg_bounded
            # )
            # loss = -loss_inv

            loss = torch.nn.functional.triplet_margin_loss(
                torch.ones_like(prob_tilde_bounded),
                prob_tilde_bounded,
                prob_tilde_neg_bounded,
            )

        else:
            log_prob = torch.log(prob_tilde) - torch.log(norm_constant)
            loss = -log_prob

        logger.debug(f"Target: {target.detach().cpu().numpy().tolist()}")
        logger.debug(f"Prob Tilde: {prob_tilde.detach().cpu().numpy().tolist()}")
        logger.debug(f"Norm Constant: {norm_constant.detach().cpu().numpy().tolist()}")
        logger.debug(f"Loss: {loss.detach().cpu().numpy().tolist()}")
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
        """Get TT parameters of joint distribution from input embeddings

        Args:
            x (torch.Tensor): Input embeddings. Shape: (B, T, D)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Alpha, beta, core. Shapes: (B, R), (B, R), (B, R, D, R)
        """

        batch_size, seq_len, _ = x.shape
        z_alpha = x[:, -1, :] @ self.w_alpha  # (B, R)
        z_beta = x[:, -1, :] @ self.w_beta  # (B, R)
        z_core = (x[:, -1, :] @ self.w_vocab).reshape(
            batch_size, self.rank, self.vocab_size, self.rank
        )  # (B, D * R * R)

        alpha = torch.abs(z_alpha)
        beta = torch.abs(z_beta)
        dcore = torch.abs(z_core) * (1 / seq_len)

        if self.mode == "tjd-ident":
            # Alter core s.t core[b, :, d, :] = I for d in identity_transform_ids
            core_ident = create_core_ident(
                batch_size=batch_size, vocab_size=self.vocab_size, rank=self.rank
            ).to(dcore.device)
            mask = torch.ones_like(core_ident).to(dcore.device)
            mask[:, :, self.identity_transform_ids, :] = 0
            core = core_ident + mask * dcore

            for tens_name, tens_val in zip(
                ["alpha", "beta", "core"], [alpha, beta, core]
            ):
                check_naninf(tens_val, f"_get_tt_params:{tens_name}")
                tens_min, tens_max, tens_mean, tens_std = (
                    torch.min(tens_val),
                    torch.max(tens_val),
                    torch.mean(tens_val),
                    torch.std(tens_val),
                )
                logger.debug(
                    f"{tens_name}: min: {tens_min:.4f} | max: {tens_max:.4f} | mean: {tens_mean:.4f} | std: {tens_std:.4f}"
                )
        else:
            core = dcore

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
        debug: bool = True,
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

        self._validate_forward_args(input_embs, label_ids)

        # Transform target using self.identity_transform_label_id_token_id
        if self.identity_transform_label_id_token_id is not None:
            # target [B, T] tensor
            # elementwise transform target[k] = self.identity_transform_label_id_token_id[target[k]] if target[k] in self.identity_transform_label_id_token_id
            y = apply_id_transform(label_ids, self.identity_transform_label_id_token_id)
        else:
            y = label_ids

        if self.mode == "ce":
            batch_size, seq_len, _ = input_embs.shape
            z_core = (
                input_embs.reshape(batch_size * seq_len, -1) @ self.w_vocab
            ).reshape(
                batch_size * seq_len, self.rank, self.vocab_size, self.rank
            )  # (B*T, R, V, R)

            z_alpha = torch.ones(batch_size * seq_len, self.rank, device=z_core.device)
            z_beta = torch.ones(batch_size * seq_len, self.rank, device=z_core.device)
            logits = torch.einsum("bi, bidj, bj->bd", z_alpha, z_core, z_beta)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), y.view(-1))
            return loss

        elif self.mode == "ce-plus":  # works
            x = input_embs
            alpha, beta, core = self._get_tt_params(x)  # (B, R), (B, R), (B, R, D, R)
            prob_tilde = torch.einsum("bi, bidj, bj->bd", alpha, core, beta)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(prob_tilde.view(-1, prob_tilde.size(-1)), y.view(-1))
            return loss

        elif self.mode == "log-softmax":
            x = input_embs
            alpha, beta, core = self._get_tt_params(x)  # (B, R), (B, R), (B, R, D, R)
            prob_tilde = torch.einsum("bi, bidj, bj->bd", alpha, core, beta)  # (B, D)
            log_prob = torch.nn.functional.log_softmax(prob_tilde, dim=-1)
            loss = -log_prob
            return loss.mean()

        elif self.mode == "log-prob":
            x = input_embs
            alpha, beta, core = self._get_tt_params(x)  # (B, R), (B, R), (B, R, D, R)
            prob_tilde = torch.einsum("bi, bidj, bj->bd", alpha, core, beta)  # (B, D)
            prob_tilde_indexed = torch.stack(
                [prob_tilde[i, y.reshape(-1)[i]] for i in range(prob_tilde.size(0))]
            )  # (B,)
            norm_constant = torch.einsum(
                "bi, bidj, bj, bd->b",
                alpha,
                core,
                beta,
                torch.ones_like(prob_tilde, device=prob_tilde.device),
            )  # (B,)
            log_prob = torch.log(prob_tilde_indexed) - torch.log(norm_constant)
            loss = -log_prob
            return loss.mean()

        else:
            x = input_embs
            alpha, beta, core = self._get_tt_params(x)  # (B, R), (B, R), (B, R, D, R)

            # Grad estimation
            # a-G-b
            if debug:
                idx = y[0, 0]
                grad_core_coeff_1 = (
                    1 / torch.einsum("bi,bidj,bj->bd", alpha, core, beta)
                )[0, idx]
                ident = torch.zeros(self.vocab_size, device=core.device)
                ident[idx] = 1
                grad_core_outer_product_1 = torch.einsum(
                    "bi,d,bj->bidj", alpha, ident, beta
                )[0]
                grad_core_1 = grad_core_coeff_1 * grad_core_outer_product_1

                ones = torch.ones(core.size(0), self.vocab_size, device=core.device)
                grad_core_coeff_2 = (
                    1 / torch.einsum("bi,bidj,bj,bd->b", alpha, core, beta, ones)[0]
                )
                grad_core_outer_product_2 = torch.einsum(
                    "bi,bd,bj->bidj", alpha, ones, beta
                )[0]
                grad_core_2 = grad_core_coeff_2 * grad_core_outer_product_2

                grad_tot = -grad_core_1 + grad_core_2
                logger.debug(f"dl/dcore (min): {torch.min(grad_tot):.4f}")
                logger.debug(f"dl/dcore (max): {torch.max(grad_tot):.4f}")
                logger.debug(f"dl/dcore (mean): {torch.mean(grad_tot):.4f}")
                logger.debug(f"dl/dcore (std): {torch.std(grad_tot):.4f}")

            for tensor, tensor_name in zip(
                [alpha, beta, core], ["alpha", "beta", "core"]
            ):
                check_naninf(tensor, f"forward:{tensor_name}")

            loss = self._compute_loss(alpha, beta, core, y)
            check_naninf(loss, f"forward:loss")
            return loss
