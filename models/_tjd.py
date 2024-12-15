import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch

from distributions.base import BaseDist
from distributions.cp import CPDist
from distributions.full import FullDist
from distributions.mps import MPSDist
from distributions.umps import UMPSDist
from tensorops.common import get_windowed_input_ids
from utils.beam_search import beam_search, get_candidates


DIST_MAP = {
    "full": FullDist,
    "cp": CPDist,
    "mps": MPSDist,
    "umps": UMPSDist,
    "base": BaseDist,
}


class TJD(ABC, torch.nn.Module):
    def __init__(
        self,
        # TODO: rename n_embd, vocab_size to embd_size, vocab_size
        n_embd,
        vocab_size,
        rank: int = 1,
        horizon: int = 1,
        positivity_func: str = "exp",
        model_head: str = "base",
        eps: float = 1e-9,
        model_kwargs: Dict = {},
    ):
        """Initialize the TJD model.

        Args:
            n_embd (int): Embedding size.
            vocab_size (int): Vocabulary size.
            rank (int, optional): Rank of the joint distribution. Defaults to 1.
            horizon (int, optional): Horizon of the joint distribution. Defaults to 1.
            positivity_func (str, optional): Positivity function. Defaults to "exp".
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-9.
            model_head (str, optional): Language model head. Defaults to "base" (i.e., no joint distribution).

        """
        super().__init__()
        self.rank = rank
        self.horizon = horizon
        self.eps = eps
        self.model = self.get_model(**model_kwargs)
        self.model_head = DIST_MAP[model_head](
            n_embd=n_embd,
            vocab_size=vocab_size,
            rank=rank,
            horizon=horizon,
            positivity_func=positivity_func,
        )
        self.vocab_size = vocab_size
        self.n_embd = n_embd

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def get_model(self, **kwargs) -> torch.nn.Module:
        """Get the torch model to be modified.

        Returns:
            torch.nn.Module: Model to be modified.
        """
        pass

    @abstractmethod
    def get_last_hidden_state(
        self, input_ids: torch.Tensor, attention_mask=None
    ) -> torch.Tensor:
        """Get the last hidden state of the model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (B, T).
            attention_mask ([type], optional): Attention mask of shape (B, T). Defaults to None.

        Returns:
            torch.Tensor: Last hidden state of shape (B, T, n_embd).
        """
        pass

    # TODO: rename to get_runtime_horizon or is_valid_horizon
    def _get_horizon(self, horizon: Optional[int]) -> int:
        """Get the horizon value. Prevents runtime horizon exceeding the model horizon.

        Args:
            horizon (Optional[int]): Candidate horizon value.

        Raises:
            ValueError: If the horizon is greater than the model horizon.

        Returns:
            int: Horizon value.
        """
        horizon = self.horizon if horizon is None else horizon
        if horizon > self.horizon:
            raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
        return horizon

    def generate(
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
            top_k (int, optional): Top k sampling. Defaults to 50.

        Returns:
            torch.Tensor: Generated tokens of shape (B, `max_new_tokens`).
        """
        assert input_ids.size(0) == 1, "Only batch size 1 is supported"

        dvc = input_ids.device
        horizon = self._get_horizon(horizon)
        last_hidden_states = [
            self.get_last_hidden_state(input_ids)[:, -1:, :]
        ] * num_beams

        def expand_fn(beams):
            nonlocal last_hidden_states  # Allow modification of outer variable
            seqs, seq_log_probs = zip(*beams)  # Lists of shape (n_beams, T), (n_beams,)
            seq_log_probs = torch.tensor(seq_log_probs).to(dvc)

            time_step = len(seqs[0])
            if time_step % horizon == 0 and time_step != 0:
                # print(f"[Hidden states] Time step: {time_step} (horizon: {horizon})")
                last_hidden_states = []
                seqs_tensor = torch.tensor(seqs).to(dvc)
                for sq in seqs_tensor:
                    inp = torch.cat([input_ids, sq.reshape(1, -1)], dim=1)
                    hidden = self.get_last_hidden_state(inp)[:, -1:, :]
                    last_hidden_states.append(hidden)

            next_token_probs = []
            for i_beam, seq in enumerate(seqs):
                sub_time_step = time_step % horizon
                sub_seq = seq[-sub_time_step:] if sub_time_step != 0 else []
                ops_tensor = torch.tensor(
                    sub_seq + [-1] + [-2] * (horizon - sub_time_step - 1),
                    device=dvc,
                )
                # NOTE: get_pos_params called many times with the same `hidden_state`
                probs_next, _ = self.model_head.get_dist(
                    hidden_state=last_hidden_states[i_beam],
                    ops=ops_tensor,
                    use_cache=False if sub_time_step == 0 else True,
                    save_cache=True,
                )  # (V,)
                assert (
                    len(probs_next.shape) == 1 and probs_next.size(0) == self.vocab_size
                ), "Invalid shape for probs_next"
                next_token_probs.append(probs_next)

            next_token_probs = torch.stack(next_token_probs)  # (n_beams, V)
            return get_candidates(
                seqs=seqs,
                seq_log_probs=seq_log_probs,
                next_token_probs=next_token_probs,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        horizon: Optional[int] = None,
        reduce="mean",
        use_memory_efficient_loss: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (B, T)
            labels (torch.Tensor): Tensor of shape (B, T)
            horizon (Optional[int], optional): Joint distribution size. If None, uses the model level horizon. Defaults to None.
            reduce (str, optional): Reduction method. Defaults to "mean".
            use_memory_efficient_loss (bool, optional): Whether to use memory efficient loss computation. Defaults to False.

        Note:
            horizon must be less than or equal to the model horizon specified during initialization.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - loss (torch.Tensor): Reduced loss value of shape (B,)
                - nll (torch.Tensor): Reduced negative log likelihood of shape (B,)
                - loss_scale (torch.Tensor): Loss scaling factor, scalar tensor of value 1/rank

        """

        # Sequence length must be greater than horizon
        assert (
            input_ids.size(1) > self.horizon
        ), "Sequence length must be greater than horizon"

        batch_size, _ = input_ids.size()
        horizon = self._get_horizon(horizon)

        last_hidden_state = self.get_last_hidden_state(
            input_ids, attention_mask=attention_mask
        )
        targets = get_windowed_input_ids(input_ids, horizon=horizon).reshape(
            batch_size, -1, horizon
        )  # (B, T-H, H)

        assert targets.size(1) >= horizon, "Invalid targets"

        last_hidden_state_ds = last_hidden_state[:, :-horizon]  # (B, T-H, D)
        targets_ds = targets  # (B, T-H, H)
        if use_memory_efficient_loss:
            # Downsample hidden states and targets
            last_hidden_state_ds = last_hidden_state_ds[:, ::horizon]
            targets_ds = targets[:, ::horizon]

        p_tilde, p_tilde_scale_factors, norm_const, norm_const_scale_factors = (
            self.model_head.evaluate_at_points_and_get_norm_consts(
                last_hidden_state_ds, targets_ds
            )
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
        # NLL computation requires only each horizon-th element
        nll = loss if use_memory_efficient_loss else loss[:, ::horizon]
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
