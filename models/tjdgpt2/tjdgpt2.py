from typing import Dict, Optional
import line_profiler
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)

from distributions.base import BaseDist
from distributions.cp import CPDist
from distributions.full import FullDist
from distributions.mps import MPSDist
from distributions.umps import UMPSDist

from tensorops.common import get_windowed_input_ids
from utils.beam_search import beam_search, get_candidates


# TODO: Apply loss scaling in the forward pass directly
class TJDGPT2(torch.nn.Module):
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
        is_full_rank: bool = False,
    ):
        super().__init__()
        self.generate = self.generateV2
        self.model_name = model
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
            "eos_token_id": eos_token_id,
            "bos_token_id": bos_token_id,
            "pad_token_id": pad_token_id,
            "is_full_rank": is_full_rank,
        }
        self.rank = rank
        self.vocab_size = vocab_size
        self.horizon = horizon
        self.eps = eps
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
        self.model_head = {
            "full": FullDist,
            "cp": CPDist,
            "mps": MPSDist,
            "umps": UMPSDist,
            "base": BaseDist,
        }[model](
            n_embd=n_embd,
            vocab_size=vocab_size,
            rank=rank,
            horizon=horizon,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_last_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
        )
        return transformer_outputs.last_hidden_state

    def _get_horizon(self, horizon: Optional[int]):
        horizon = self.horizon if horizon is None else horizon
        if horizon > self.horizon:
            raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
        return horizon

    @line_profiler.profile
    def generateV2(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 8,
        num_beams: int = 1,
        do_sample: bool = True,
        horizon: Optional[int] = None,
        top_k: int = 50,
        **kwargs,
    ):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.
            num_beams (int, optional): Number of beams. Defaults to 1.
            do_sample (bool, optional): Whether to sample. Defaults to False.
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.

        Returns:
            torch.Tensor: Generated tokens of shape (B, `max_new_tokens`).
        """
        assert input_ids.size(0) == 1, "Only batch size 1 is supported"

        dvc = input_ids.device
        horizon = self._get_horizon(horizon)
        last_hidden_states = [
            self._get_last_hidden_state(input_ids)[:, -1:, :]
        ] * num_beams

        @line_profiler.profile
        def expand_fn(beams):
            nonlocal last_hidden_states  # Allow modification of outer variable
            seqs, log_probs = zip(*beams)  # Lists of shape (n_beams, T), (n_beams,)
            log_probs = torch.tensor(log_probs).to(dvc)

            time_step = len(seqs[0])
            if time_step % horizon == 0 and time_step != 0:
                # print(f"[Hidden states] Time step: {time_step} (horizon: {horizon})")
                last_hidden_states = []
                seqs_tensor = torch.tensor(seqs).to(dvc)
                for sq in seqs_tensor:
                    inp = torch.cat([input_ids, sq.reshape(1, -1)], dim=1)
                    hidden = self._get_last_hidden_state(inp)[:, -1:, :]
                    last_hidden_states.append(hidden)

            all_probs_next = []
            for i_beam, seq in enumerate(seqs):
                sub_time_step = time_step % horizon
                sub_seq = seq[-sub_time_step:] if sub_time_step != 0 else []
                ops_tensor = torch.tensor(
                    sub_seq + [-1] + [-2] * (horizon - sub_time_step - 1),
                    device=dvc,
                )
                # BUG: get_pos_params called many times with the same `hidden_state`
                probs_next, _ = self.model_head.get_dist(
                    hidden_state=last_hidden_states[i_beam],
                    ops=ops_tensor,
                    use_cache=False if sub_time_step == 0 else True,
                    save_cache=True,
                )  # (V,)
                assert (
                    len(probs_next.shape) == 1 and probs_next.size(0) == self.vocab_size
                ), "Invalid shape for probs_next"
                all_probs_next.append(probs_next)

            all_probs_next = torch.stack(all_probs_next)  # (n_beams, V)
            return get_candidates(
                seqs=seqs,
                log_probs=log_probs,
                next_token_log_probs=all_probs_next,
                num_beams=num_beams,
                do_sample=do_sample,
                top_k=top_k,
            )

        # Run beam search
        best_seq, _ = beam_search(
            expand_fn=expand_fn,
            initial_beam=[([], 0.0)],
            num_beams=num_beams,
            max_steps=max_new_tokens,
        )
        return torch.tensor(best_seq, device=dvc).reshape(1, -1)

    @line_profiler.profile
    def generateV1(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 8,
        num_beams=1,
        do_sample=False,
        horizon: Optional[int] = None,
        **kwargs,
    ):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.
            num_beams (int, optional): Number of beams. Defaults to 1.
            do_sample (bool, optional): Whether to sample. Defaults to False.
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.

        Returns:
            torch.Tensor: Generated tokens of shape (B, `max_new_tokens`).
        """
        assert input_ids.size(0) == 1, "Only batch size 1 is supported"
        horizon = self._get_horizon(horizon)
        batch_size, _ = input_ids.size()
        n_passes = max(max_new_tokens // horizon, 1)
        output_tens = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        input_tens = input_ids

        for _ in range(n_passes):
            last_hidden_state = self._get_last_hidden_state(input_tens)
            sample = self.model_head.generate(
                last_hidden_state=last_hidden_state, horizon=horizon
            )  # (B, H)
            output_tens = torch.cat([output_tens, sample], dim=1)
            input_tens = torch.cat([input_tens, sample], dim=1)
        return output_tens

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        horizon: Optional[int] = None,
        reduce="mean",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (B, T)
            labels (torch.Tensor): Tensor of shape (B, T)
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.

        Note:
            The horizon applied in the forward pass is the minimum of the model level horizon and the horizon passed as an argument.

        Returns:
            torch.Tensor: Loss value.
        """

        # Sequence length must be greater than horizon
        assert (
            input_ids.size(1) > self.horizon
        ), "Sequence length must be greater than horizon"

        batch_size, _ = input_ids.size()
        horizon = self._get_horizon(horizon)

        last_hidden_state = self._get_last_hidden_state(input_ids)
        targets = get_windowed_input_ids(input_ids, horizon=horizon).reshape(
            batch_size, -1, horizon
        )  # (B, T-H, H)

        assert targets.size(1) >= horizon, "Invalid targets"

        p_tilde, p_tilde_scale_factors = self.model_head.evaluate_at_points(
            last_hidden_state[:, :-horizon], targets
        )  # (B, T-H)

        norm_const, norm_const_scale_factors = self.model_head.get_norm_consts(
            last_hidden_state[:, :-horizon], horizon=horizon
        )  # (B, T-H)

        # Health checks
        # 1. Ensure no NaNs
        assert not torch.isnan(p_tilde).any(), "p_tilde NaN"
        assert not torch.isnan(norm_const).any(), "norm_const NaN"
        # 2. Ensure p_tilde < norm_const (if no scale factors)
        if len(p_tilde_scale_factors) == 0 and len(norm_const_scale_factors) == 0:
            assert (p_tilde < norm_const).all(), "p_tilde < norm_const"

        loss = (
            -torch.log(p_tilde + self.eps)
            + torch.log(norm_const)
            # Contraction Stability Scale Factors
            - sum([torch.log(z) for z in p_tilde_scale_factors])
            + sum([torch.log(z) for z in norm_const_scale_factors])
        )  # (B, T-H)

        # Train loss
        nll = loss[:, ::horizon]  # batch and seq mean of negative log likelihood
        reduct_fn = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x,
        }[reduce]
        return {
            "loss": reduct_fn(loss.sum(dim=-1)),
            "nll": reduct_fn(nll.sum(dim=-1)),
            "loss_scale": torch.tensor(1 / self.rank).to(loss.device),
        }
