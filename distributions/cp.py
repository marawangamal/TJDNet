from typing import List, Tuple
import torch
from TJDNet.TGPT2 import cp_outer_product
from utils.tensop import batch_multi_dim_index, sample_from_tens


class CPDist(torch.nn.Module):
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
        self.horizon = horizon
        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[positivity_func]

    def _get_materialized_dist(self, last_hidden_state: torch.Tensor, horizon: int):
        assert (
            horizon == None or horizon <= self.horizon
        ), "Horizon must be <= model horizon"
        horizon = horizon if horizon is not None else self.horizon
        params = self.positivity_func(
            self.param_func(last_hidden_state[:, -1:, :])
        )  # (B, 1, V*R*H) we only need the Tth hidden state
        p_tilde = cp_outer_product(
            params.reshape(-1, self.horizon, self.vocab_size, self.rank)
        )  # (B, V**H)
        p_tilde = p_tilde.reshape(
            -1, *([self.vocab_size] * horizon)
        )  # (B, V, V, ..., V)
        return p_tilde

    def generate(self, last_hidden_state: torch.Tensor, horizon: int):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            horizon (int): Horizon of the generation (Must be <= Horizon of the model)
        """
        # Cannot generate sequences longer than `horizon`
        p_tilde = self._get_materialized_dist(
            last_hidden_state, horizon
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
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, H, D)
            is_normalized (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points. Shape (B, H)
        """
        # return torch.abs(torch.rand(points.size(0), points.size(1))), []
        horizon = points.size(1)
        p_tilde = self._get_materialized_dist(
            last_hidden_state, horizon
        )  # (B, V, V, ..., V)
        return batch_multi_dim_index(p_tilde, points), []  # (B,)

    def get_norm_const(
        self, last_hidden_state: torch.Tensor, horizon: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        p_tilde = self._get_materialized_dist(
            last_hidden_state, horizon
        )  # (B, V, V, ..., V)
        return p_tilde.reshape(p_tilde.size(0), -1).sum(dim=-1), []  # (B,)
