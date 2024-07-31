from typing import Any, List, Optional, Tuple, Dict, Union
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import init

import logging

from TJDNet.TJDLayer.TTDist import TTDist
from TJDNet.TJDLayer.utils import check_naninf

from .utils import create_core_ident, apply_id_transform

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs


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
