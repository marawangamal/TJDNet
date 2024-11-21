from typing import List, Tuple
import torch

from distributions.base import BaseDistribution
from utils.tensop import batch_multi_dim_index, cp_outer_product, sample_from_tens


# class CPNetwork:
#     def __init__():
#         pass


class CPDist(BaseDistribution):
    def __init__(
        self,
        n_embd: int,
        vocab_size,
        rank: int,
        horizon: int,
        positivity_func: str = "exp",
    ):
        """CP Distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        super().__init__()
        assert horizon == 2, "Only horizon=2 is supported for now"
        self.param_func = torch.nn.Linear(n_embd, vocab_size * rank * horizon)
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.rank = rank
        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[positivity_func]

    def _get_last_conditional_dist(self, last_hidden_state: torch.Tensor):
        params = self.positivity_func(
            self.param_func(last_hidden_state[:, -1:, :])
        )  # (B, 1, V*R*H) we only need the Tth hidden state
        p_tilde = cp_outer_product(
            params.reshape(-1, self.horizon, self.vocab_size, self.rank)
        )  # (B, V**H)
        p_tilde = p_tilde.reshape(
            -1, *([self.vocab_size] * self.horizon)
        )  # (B, V, V, ..., V)
        return p_tilde

    def _get_conditional_dists(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Get the distribution for the next `horizon` tokens.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            horizon (int): Number of tokens to predict. Must be <= model horizon

        Returns:
           torch.Tensor: Distribution of the next `horizon` tokens. Shape (B, V, V, ..., V)
        """
        batch_size, seq_len, _ = last_hidden_state.size()
        params = self.positivity_func(
            self.param_func(last_hidden_state)
        )  # (B, T, V*R*H) we need all the hidden states
        p_tilde = cp_outer_product(
            params.reshape(-1, self.horizon, self.vocab_size, self.rank)
        )  # (BT, V**H)
        p_tilde = p_tilde.reshape(
            batch_size, seq_len, *([self.vocab_size] * self.horizon)
        )
        return p_tilde  # (B, T, V, V, ..., V)

    def generate(self, last_hidden_state: torch.Tensor):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            horizon (int): Horizon of the generation (Must be <= Horizon of the model)
        """
        # Cannot generate sequences longer than `horizon`
        p_tilde = self._get_last_conditional_dist(
            last_hidden_state
        )  # (B, V, V, ..., V)
        return sample_from_tens(p_tilde, 1)  # (B, H)

    def evaluate_at_points(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        is_normalized=False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, H)
            is_normalized (bool, optional): Whether the points are normalized. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points of Shape (B, H) and scale_tensors (empty list)
        """
        p_tilde = self._get_conditional_dists(last_hidden_state)[
            :, : -self.horizon, ...
        ]  # (B, T-H, V, V, ..., V)
        p_tilde_reshaped = p_tilde.reshape(
            -1, *([self.vocab_size] * self.horizon)
        )  # (B*(T-H), V, V, ..., V)
        return batch_multi_dim_index(p_tilde_reshaped, points), []  # (B*(T-H))

    def get_norm_consts(
        self, last_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len, _ = last_hidden_state.size()
        p_tilde = self._get_conditional_dists(last_hidden_state)[
            :, : -self.horizon, ...
        ]  # (B, T-H, V, V, ..., V)
        return (
            p_tilde.reshape(batch_size * (seq_len - self.horizon), -1).sum(
                dim=-1
            ),  # (B*(T-H))
            [],
        )
