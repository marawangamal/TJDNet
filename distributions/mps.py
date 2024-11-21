from typing import List, Tuple
import torch
from distributions.base import BaseDistribution
from utils.tensop import sample_from_tens
from utils.tensorops.mps import (
    umps_materialize_batched,
    umps_select_marginalize_batched,
)


class MPSDist(BaseDistribution):
    def __init__(
        self,
        n_embd: int,
        vocab_size,
        rank: int,
        horizon: int,
        positivity_func: str = "exp",
    ):

        self.rank = rank
        self.vocab_size = vocab_size
        self.horizon = horizon
        self.positivity_func: torch.nn.Module = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[positivity_func]
        self.tensor_train_size = rank + (rank * vocab_size * rank) + rank
        self.param_func = torch.nn.Linear(n_embd, self.tensor_train_size)

    def _get_pos_params(self, last_hidden_state: torch.Tensor):
        params = self.positivity_func(
            self.param_func(last_hidden_state)
        )  # (B, T, R*H*V)
        alpha, core, beta = torch.split(
            params,
            [self.rank, self.rank * self.vocab_size * self.rank, self.rank],
            dim=-1,
        )  # (B, T, R), (B, T, R*V*R), (B, T, R)
        return alpha, core, beta

    def generate(self, last_hidden_state: torch.Tensor):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            horizon (int): Horizon of the generation (Must be <= Horizon of the model)
        """
        # Cannot generate sequences longer than `horizon`
        alpha, core, beta = self._get_pos_params(
            last_hidden_state[:, -1:, :]
        )  # (B, 1, V*R*H) we only need the Tth hidden state
        p_tilde = umps_materialize_batched(
            alpha=alpha,
            beta=beta,
            core=core,
            n_core_repititions=self.horizon,
        )  # (B, V, V, ..., V)  `horizon` times
        return sample_from_tens(p_tilde[0], 1)  # (B, H)

    def evaluate_at_points(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        is_normalized=False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, H)
            is_normalized (bool, optional): Whether the points are normalized. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points of Shape (B, H) and scale_tensors (empty list)
        """
        batch_size, seq_len, _ = last_hidden_state.shape
        alpha, core, beta = self._get_pos_params(last_hidden_state)
        # (B, T, R*H*V) => (B, T)
        p_tilde, scale_factors = umps_select_marginalize_batched(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len, self.rank, self.vocab_size, self.rank
            ),
            operation_map=points.reshape(batch_size * seq_len, -1),
        )  # (batch_size, n_vocab)
        return p_tilde.reshape(batch_size, seq_len), scale_factors

    def get_norm_consts(
        self, last_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get the normalization constants for the BT distributions.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Norm constants and scale tensors
        """
        raise NotImplementedError
