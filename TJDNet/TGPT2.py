from typing import List, Tuple
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)

from .MPSDist import MPSDistBase
from .loss import get_entropy_loss_stable, get_entropy_loss_stable_mjd
from .utils import (
    window_input_ids,
    AverageMeter,
    get_windowed_input_ids,
)
from ..utils.tensop import sample_from_tens, batch_multi_dim_index

# TODO:
# Refactor to have only `TJDGPT2`, and it accepts as input a decendent of the `JDist` class that has, param_count, sample and evalute methods


class JDist(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 8, horizon=None, **kwargs
    ):
        raise NotImplementedError


class GPT2(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        eos_token_id: int = 50256,
        bos_token_id: int = 50256,
        pad_token_id: int = 50256,
        **kwargs,
    ):
        super().__init__()
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=vocab_size,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                dropout=dropout,
                eos_token_id=eos_token_id,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
            )
        )

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def forward(self, input_ids, labels, horizon=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            **kwargs,
        )


class TJDGPT2OLD(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        rank: int = 2,
        eps: float = 1e-9,
        horizon: int = 8,
        positivity_func: str = "sq",
        eos_token_id: int = 50256,
        bos_token_id: int = 50256,
        pad_token_id: int = 50256,
        **kwargs,
    ):
        super().__init__()
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=vocab_size,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                dropout=dropout,
                eos_token_id=eos_token_id,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
            )
        )

        self.positivity_func = positivity_func

        self.rank = rank

        self.vocab_size = vocab_size
        self.eps = eps
        self.custom_unembedding = torch.nn.Linear(n_embd, vocab_size, bias=False)
        self.tensor_train_size = rank + (rank * vocab_size * rank) + rank
        self.seq2latents = torch.nn.Sequential(
            # Average pool the seq_len dimension
            torch.nn.Linear(n_embd, n_embd),
            torch.nn.ReLU(),
        )
        self.latent2tt = torch.nn.Linear(n_embd, self.tensor_train_size)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.horizon = horizon

        self.forward_avg_meter = AverageMeter()
        self.loss_avg_meter = AverageMeter()

    @property
    def device(self):
        return next(self.parameters()).device

    # TODO: use delta_core
    def _get_tt_dist(self, input_ids: torch.Tensor, horizon: int, **kwargs):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            **kwargs,
        )

        hidden_states = transformer_outputs.last_hidden_state
        alpha, beta, core = self._get_tt_params(
            hidden_states[:, :-horizon, :]
        )  # (B, T-H, R), (B, T-H, R), (B, T-H, R, D, R)
        batch_size, seq_len_adj, rank, vocab_size, _ = core.size()

        # Forward pass:
        learned_mpsdist = MPSDistBase(
            alpha.reshape(batch_size * seq_len_adj, -1),
            beta.reshape(batch_size * seq_len_adj, -1),
            core.reshape(batch_size * seq_len_adj, rank, vocab_size, rank),
            positivity_func=self.positivity_func,
        )

        # 1. Window the `input_ids` to get targets: (B, T) => (B, T, H)
        #   each position should look H steps ahead
        input_ids_windowed = window_input_ids(input_ids, horizon=horizon)

        # 2. Make targets using windowed input_ids
        targets = input_ids_windowed[:, :-horizon]  # (B, T-H, H)
        targets = targets.reshape(-1, horizon)  # (B * (T-H), H)

        return learned_mpsdist, transformer_outputs, targets

    def _get_tt_params(self, hidden_states: torch.Tensor):
        # Map with linear layer
        batch_size, seq_len, hidden_size = hidden_states.size()
        tt_latent = self.seq2latents(
            hidden_states
        )  # (batch_size, seq_len, hidden_size)
        tt_params = self.latent2tt(
            tt_latent
        )  # (batch_size, seq_len, tensor_train_size)
        alpha, core, beta = torch.split(
            tt_params,
            [self.rank, self.rank * self.vocab_size * self.rank, self.rank],
            dim=-1,
        )
        alpha = alpha.reshape(batch_size, seq_len, self.rank)
        beta = beta.reshape(batch_size, seq_len, self.rank)
        core = core.reshape(batch_size, seq_len, self.rank, self.vocab_size, self.rank)
        return alpha, beta, core

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 8, horizon=None, **kwargs
    ):
        """Generate new tokens given an input prompt.

        Args:
            input_ids (torch.Tensor): Input prompt tensor of shape (B, T).
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.

        Returns:
            torch.Tensor: Generated tokens of shape (B, max_new_tokens).
        """

        batch_size, seq_len = input_ids.size()
        horizon = horizon if horizon is not None else self.horizon
        n_passes = max_new_tokens // horizon
        output_tens = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        input_tens = input_ids

        for _ in range(n_passes):
            transformer_outputs = self.model.transformer(
                input_ids=input_tens,
            )
            hidden_states = transformer_outputs.last_hidden_state

            alpha, beta, core = self._get_tt_params(
                hidden_states[:, -1:, :]
            )  # (B, 1, R, D, R)
            _, seq_len_adj, rank, vocab_size, _ = core.size()

            # Forward pass:
            learned_mpsdist = MPSDistBase(
                alpha.reshape(batch_size * seq_len_adj, -1),
                beta.reshape(batch_size * seq_len_adj, -1),
                core.reshape(batch_size * seq_len_adj, rank, vocab_size, rank),
                positivity_func=self.positivity_func,
            )

            sample = learned_mpsdist.sample(max_len=horizon)  # (B, H)
            output_tens = torch.cat([output_tens, sample], dim=1)
            input_tens = torch.cat([input_tens, sample], dim=1)
        return output_tens

    def forward(self, input_ids, labels=None, horizon=None, **kwargs):
        learned_mpsdist, transformer_outputs, targets = self._get_tt_dist(
            input_ids,
            horizon=horizon if horizon is not None else self.horizon,
            **kwargs,
        )
        loss = get_entropy_loss_stable(
            learned_mpsdist,
            targets=targets,
            eps=self.eps,
        )
        transformer_outputs.loss = loss
        return transformer_outputs


# TODO: Rename to MJDGPT2
class MGPT2(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        eos_token_id: int = 50256,
        bos_token_id: int = 50256,
        pad_token_id: int = 50256,
        horizon: int = 2,
        positivity_func: str = "exp",
        **kwargs,
    ):
        super().__init__()
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=vocab_size,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                dropout=dropout,
                eos_token_id=eos_token_id,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
            )
        )
        self.latent2mjd = torch.nn.Linear(n_embd, vocab_size**horizon)
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[positivity_func]

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _get_windowed_input_ids(input_ids: torch.Tensor, horizon: int):
        # 1. Window the `input_ids` to get targets: (B, T) => (B, T, H)
        #   each position should look H steps ahead
        input_ids_windowed = window_input_ids(input_ids, horizon=horizon)

        # 2. Make targets using windowed input_ids
        targets = input_ids_windowed[:, :-horizon]  # (B, T-H, H)
        targets = targets.reshape(-1, horizon)  # (B * (T-H), H)
        return targets

    def _get_preds(self, input_ids: torch.Tensor, **kwargs):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        mjdist_flat = self.positivity_func(
            self.latent2mjd(hidden_states)
        )  # (B, T, V**H)
        mjdist = mjdist_flat[:, : -self.horizon, :].reshape(
            -1, *([self.vocab_size] * self.horizon)
        )
        return mjdist, transformer_outputs

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, **kwargs):
        """Generate new tokens given an input prompt.

        Args:
            input_ids (torch.Tensor): Input prompt tensor of shape (B, T).
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.

        Returns:
            torch.Tensor: Generated tokens of shape (B, max_new_tokens).
        """

        batch_size, _ = input_ids.size()
        n_passes = max_new_tokens // self.horizon
        output_tens = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        input_tens = input_ids

        for _ in range(n_passes):
            transformer_outputs = self.model.transformer(
                input_ids=input_tens,
            )
            hidden_states = transformer_outputs.last_hidden_state
            mjdist_flat = self.positivity_func(
                self.latent2mjd(hidden_states[:, -1:, :])
            )  # (B, 1, V**H)
            mjdist = mjdist_flat.reshape(-1, *([self.vocab_size] * self.horizon))
            sample = sample_from_tens(mjdist, 1)
            output_tens = torch.cat([output_tens, sample], dim=1)
            input_tens = torch.cat([input_tens, sample], dim=1)
        return output_tens

    def forward(self, input_ids, labels=None, horizon=None, **kwargs):
        mjdist, transformer_outputs = self._get_preds(input_ids, **kwargs)
        targets = self._get_windowed_input_ids(
            input_ids, horizon=self.horizon
        )  # (B * T-H, H)
        loss = get_entropy_loss_stable_mjd(
            mjdist,
            targets=targets,
        )
        transformer_outputs.loss = loss
        return transformer_outputs


# TODO: Rename to MJDGPT2


def latent_to_cp_mjd(
    x: torch.Tensor,
    cp_params_func: torch.nn.Linear,
    horizon: int,
    vocab_size: int,
    rank: int,
):
    """_summary_

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, n_emb)
        cp_params_func (torch.Tensor): CP Parameter projeciton n_emb => vocab_size*rank*horizon
        horizon (int): _description_
        vocab_size (int): _description_
        rank (int): _description_

    Returns:
        torch.Tensor: matrix joint distribution of shape (B, T, V**H)
    """
    batch_size, seq_len, _ = x.size()
    cp_params = cp_params_func(x)  # (B, T, R*V*H)
    result = cp_outer_product(
        cp_params.reshape(batch_size * seq_len, horizon, vocab_size, rank)
    )  # (B*T, V**H)
    return result.reshape(batch_size, seq_len, -1)


class CPGPT2(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        eos_token_id: int = 50256,
        bos_token_id: int = 50256,
        pad_token_id: int = 50256,
        horizon: int = 2,
        rank: int = 2,
        positivity_func: str = "exp",
        **kwargs,
    ):
        super().__init__()
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=vocab_size,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                dropout=dropout,
                eos_token_id=eos_token_id,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
            )
        )
        self.cp_params_func = torch.nn.Linear(n_embd, vocab_size * rank * horizon)
        self.latent2mjd = lambda x: latent_to_cp_mjd(
            x, self.cp_params_func, horizon, vocab_size, rank
        )
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[positivity_func]

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _get_windowed_input_ids(input_ids: torch.Tensor, horizon: int):
        # 1. Window the `input_ids` to get targets: (B, T) => (B, T, H)
        #   each position should look H steps ahead
        input_ids_windowed = window_input_ids(input_ids, horizon=horizon)

        # 2. Make targets using windowed input_ids
        targets = input_ids_windowed[:, :-horizon]  # (B, T-H, H)
        targets = targets.reshape(-1, horizon)  # (B * (T-H), H)
        return targets

    def _get_preds(self, input_ids: torch.Tensor, **kwargs):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        mjdist_flat = self.positivity_func(
            self.latent2mjd(hidden_states)
        )  # (B, T, V**H)
        mjdist = mjdist_flat[:, : -self.horizon, :].reshape(
            -1, *([self.vocab_size] * self.horizon)
        )
        return mjdist, transformer_outputs

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, **kwargs):
        """Generate new tokens given an input prompt.

        Args:
            input_ids (torch.Tensor): Input prompt tensor of shape (B, T).
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.

        Returns:
            torch.Tensor: Generated tokens of shape (B, max_new_tokens).
        """

        batch_size, _ = input_ids.size()
        n_passes = max_new_tokens // self.horizon
        output_tens = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        input_tens = input_ids

        for _ in range(n_passes):
            transformer_outputs = self.model.transformer(
                input_ids=input_tens,
            )
            hidden_states = transformer_outputs.last_hidden_state
            mjdist_flat = self.positivity_func(
                self.latent2mjd(hidden_states[:, -1:, :])
            )  # (B, 1, V**H)
            mjdist = mjdist_flat.reshape(-1, *([self.vocab_size] * self.horizon))
            sample = sample_from_tens(mjdist, 1)
            output_tens = torch.cat([output_tens, sample], dim=1)
            input_tens = torch.cat([input_tens, sample], dim=1)
        return output_tens

    def forward(self, input_ids, labels=None, horizon=None, **kwargs):
        mjdist, transformer_outputs = self._get_preds(input_ids, **kwargs)
        targets = get_windowed_input_ids(
            input_ids, horizon=self.horizon
        )  # (B * T-H, H)
        loss = get_entropy_loss_stable_mjd(
            mjdist,
            targets=targets,
        )
        transformer_outputs.loss = loss
        return transformer_outputs


# class TJDHead(torch.nn.Module):
#     def __init__(
#         self,
#         n_embd: int,
#         vocab_size: int,
#         rank: int,
#         horizon: int,
#         head_type: str,
#         eps: float = 1e-6,
#         **kwargs,
#     ):
#         self.eps = eps
#         self.dist = {
#             "cp": CPDist,
#         }[head_type](
#             n_embd=n_embd,
#             vocab_size=vocab_size,
#             rank=rank,
#             horizon=horizon,
#         )

#     def generate(
#         self, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, horizon: int
#     ):
#         return self.dist.generate(input_ids, horizon=horizon)

#     def forward(
#         self, last_hidden_state: torch.Tensor, targets: torch.Tensor, horizon: int
#     ):
#         """Forward pass of CP Joint Distribution Head.

#         Args:
#             last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
#             input_ids (torch.Tensor): Input tensor of shape (B, T)
#             horizon (int): Horizon of the model (must be <= Horizon of the model)
#         """
#         assert (
#             horizon == None or horizon <= self.horizon
#         ), "Horizon must be <= model horizon"
#         horizon = horizon if horizon is not None else self.horizon
#         p_tilde, p_tilde_scale_factors = self.dist.evaluate_at_points(
#             last_hidden_state, targets
#         )
#         norm_const, norm_const_scale_factors = self.dist.get_norm_const(
#             last_hidden_state, targets
#         )
#         loss = (
#             -torch.log(p_tilde + self.eps)
#             + torch.log(norm_const)
#             - sum([torch.log(z) for z in p_tilde_scale_factors])
#             + sum([torch.log(z) for z in norm_const_scale_factors])
#         ).mean()
#         return loss


# ---------------------------------------------------------------------
# NEW
# ---------------------------------------------------------------------


# class Materialized2Dist(torch.nn.Module):
#     def __init__(self, n_embd: int, vocab_size, rank: int, horizon: int):
#         """CP Distribution

#         Args:
#             n_embd (int): Embedding dimension
#             vocab_size (int): Vocabulary size
#             rank (int): Rank of the CP decomposition
#             horizon (int): Horizon of the model (Number of tokens to predict)
#         """
#         super().__init__()
#         assert horizon == 2, "Only horizon=2 is supported for now"
#         self.param_func = torch.nn.Linear(n_embd, vocab_size * vocab_size)
#         self.vocab_size = vocab_size
#         self.rank = rank
#         self.horizon = horizon

#     def _get_materialized_dist(self, last_hidden_state: torch.Tensor, horizon: int):
#         assert (
#             horizon == None or horizon <= self.horizon
#         ), "Horizon must be <= model horizon"
#         horizon = horizon if horizon is not None else self.horizon
#         params = self.param_func(
#             last_hidden_state[:, -1:, :]
#         )  # (B, 1, V*V) we only need the Tth hidden state
#         return p_tilde

#     def generate(self, last_hidden_state: torch.Tensor, horizon: int):
#         """Generate sequences given an input tensor.

#         Args:
#             input_ids (torch.Tensor): Previous tokens of shape (B, T)
#             horizon (int): Horizon of the generation (Must be <= Horizon of the model)
#         """
#         # Cannot generate sequences longer than `horizon`
#         p_tilde = self._get_materialized_dist(
#             last_hidden_state, horizon
#         )  # (B, V, V, ..., V)
#         return sample_from_tens(p_tilde, 1)  # (B, H)

#     def evaluate_at_points(
#         self,
#         last_hidden_state: torch.Tensor,
#         points: torch.Tensor,
#         is_normalized=False,
#     ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
#         """Evaluate the distribution at the given points.

#         Args:
#             last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
#             points (torch.Tensor): Points to evaluate the distribution. Shape (B, H, D)
#             is_normalized (bool, optional): _description_. Defaults to False.

#         Returns:
#             Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points. Shape (B, H)
#         """
#         # return torch.abs(torch.rand(points.size(0), points.size(1))), []
#         horizon = points.size(1)
#         p_tilde = self._get_materialized_dist(
#             last_hidden_state, horizon
#         )  # (B, V, V, ..., V)
#         return batch_multi_dim_index(p_tilde, points), []  # (B,)

#     def get_norm_const(
#         self, last_hidden_state: torch.Tensor, horizon: torch.Tensor
#     ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
#         p_tilde = self._get_materialized_dist(
#             last_hidden_state, horizon
#         )  # (B, V, V, ..., V)
#         return p_tilde.reshape(p_tilde.size(0), -1).sum(dim=-1), []  # (B,)
