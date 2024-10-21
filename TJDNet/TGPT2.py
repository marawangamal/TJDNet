import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)

from .MPSDist import MPSDistBase
from .loss import get_entropy_loss_stable, get_entropy_loss_stable_debug
from .utils import window_input_ids, AverageMeter


class TGPT2(torch.nn.Module):
    def __init__(
        self,
        model: str = "gpt2",
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
    ):
        super().__init__()
        self.model_name = model
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
        self.rank = rank
        self.positivity_func = positivity_func
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

        self.model_config = {
            "model": model,
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
            "dropout": dropout,
            "rank": rank,
            "eps": eps,
            "horizon": horizon,
            "positivity_func": positivity_func,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    # todo: use delta_core
    def _get_tt_dist(self, input_ids: torch.Tensor, **kwargs):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            **kwargs,
        )

        hidden_states = transformer_outputs.last_hidden_state
        alpha, beta, core = self._get_tt_params(
            hidden_states[:, : -self.horizon, :]
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
        input_ids_windowed = window_input_ids(input_ids, horizon=self.horizon)

        # 2. Make targets using windowed input_ids
        targets = input_ids_windowed[:, : -self.horizon]  # (B, T-H, H)
        targets = targets.reshape(-1, self.horizon)  # (B * (T-H), H)

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

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, **kwargs):
        """Generate new tokens given an input prompt.

        Args:
            input_ids (torch.Tensor): Input prompt tensor of shape (B, T).
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.

        Returns:
            torch.Tensor: Generated tokens of shape (B, max_new_tokens).
        """

        if self.model_name == "gpt2":
            return self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        batch_size, seq_len = input_ids.size()
        n_passes = max_new_tokens // self.horizon
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

            sample = learned_mpsdist.sample(max_len=self.horizon)  # (B, H)
            output_tens = torch.cat([output_tens, sample], dim=1)
            input_tens = torch.cat([input_tens, sample], dim=1)
        return output_tens

    def forward(self, input_ids, labels, *args, **kwargs):

        if self.model_name == "gpt2":
            return self.model(
                *args,
                input_ids=input_ids,
                labels=labels,
                **kwargs,
            )

        learned_mpsdist, transformer_outputs, targets = self._get_tt_dist(
            input_ids, **kwargs
        )
        loss = get_entropy_loss_stable(
            learned_mpsdist,
            targets=targets,
            eps=self.eps,
        )

        # DEBUG: entropy loss works so no issues with `alpha`, `beta`, `core` and sampling
        # probs_tilde = learned_mpsdist.materialize(
        #     n_core_repititions=self.horizon, normalize=False
        # )
        # loss = get_entropy_loss_stable_debug(
        #     probs_tilde,
        #     targets.flatten(),
        # )
        transformer_outputs.loss = loss

        torch.cuda.synchronize()
        return transformer_outputs
