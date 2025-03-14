from typing import List, Tuple
from git import Optional
import torch
import torch.autograd.profiler as profiler

from tjdnet.distributions._base import BaseDistConfig, BaseDistribution
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched, sum_cp_tensor
from tjdnet.utils import get_positional_encodings, sample_topk


# MAT SIZES
# CP:
# - tp_net: (d_in x rank x vocab_size x horizon) (e.g  768 x 32 x 50257 x 8)

# UCP
# - tp_net: (d_in x rank x vocab_size) (e.g  768 x 32 x 50257)
# - pos_encodings: (horizon x vocab_size)  (e.g 8 x 50257)


class UCPDist(BaseDistribution):
    def __init__(self, config: BaseDistConfig, **kwargs):
        """CP Distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        config.param_net.out_dim = config.rank * config.vocab_size
        super().__init__(config)
        self.use_positional_encoding = config.use_positional_encoding

    def _get_params(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ):
        batch_size, seq_len, _ = last_hidden_state.size()  # (B, T, D)
        params = self.param_func(last_hidden_state)
        params_reshaped = params.reshape(
            batch_size, seq_len, self.rank, 1, self.vocab_size
        )

        # Add positional encoding across the horizon dimension
        pos_encodings = get_positional_encodings(
            seq_len=self.horizon, d_model=self.vocab_size, device=params.device
        )  # (H, V)
        pos_encodings = pos_encodings.reshape(
            1, 1, 1, self.horizon, self.vocab_size
        ).expand(batch_size, seq_len, self.rank, self.horizon, self.vocab_size)
        params_reshaped = params_reshaped + pos_encodings  # (B, T, R, H, V)

        if horizon is not None:
            return params_reshaped[:, :, :, :horizon, :]  # (B, T, R, H, V)

        return params_reshaped  # (B, T, R, H*, V)  // H* is model level horizon

    def sample(
        self,
        hidden_state: torch.Tensor,
        horizon: Optional[int] = None,
        do_sample: bool = False,
        top_k: int = 200,
        **kwargs,
    ) -> torch.Tensor:
        horizon = self._get_horizon(horizon)
        batch_size = hidden_state.size(0)
        dvc = hidden_state.device
        y_hat = torch.empty(batch_size, 0, device=dvc, dtype=torch.long)
        model_head_params = self._get_params(hidden_state[:, -1:, :]).squeeze(
            1
        )  # (B, 1, R, V) => (B, R, V)
        for h in range(horizon):
            ops_tensor = torch.cat(
                (
                    y_hat,  # selection
                    -1  # free leg
                    * torch.ones(batch_size, 1, dtype=torch.long, device=dvc),
                    -2  # marginalization
                    * torch.ones(
                        batch_size, (horizon - h - 1), dtype=torch.long, device=dvc
                    ),
                ),
                dim=1,
            )  # (B, T)
            p_ops_tilde, _ = select_margin_cp_tensor_batched(
                cp_params=model_head_params,
                ops=ops_tensor,
            )  # (B, V), (B, T)
            if do_sample:
                next_token = sample_topk(p_ops_tilde, top_k, num_samples=1)
            else:  # greedy sampling
                next_token = sample_topk(p_ops_tilde, 1, num_samples=1)
            y_hat = torch.cat([y_hat, next_token], dim=1)
        return y_hat  # (B, H)

    def evaluate_at_points(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, H)
            horizon (int, optional): Number of steps to consider. Defaults to model horizon.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points of Shape (B, H) and scale_tensors (empty list)
        """
        # Get indexed distribution
        batch_size, seq_len, _ = last_hidden_state.size()
        horizon = points.size(-1)
        params = self._get_params(last_hidden_state)  # (B, T, R, H, V)
        # (B, T, R, H, V) => (B, T)
        # p_tilde = select_from_cp_tensor(
        #     params.reshape(
        #         batch_size * seq_len, self.rank, horizon, self.vocab_size
        #     ),
        #     points.reshape(batch_size * seq_len, horizon),
        # )
        p_tilde, _ = select_margin_cp_tensor_batched(
            cp_params=params.reshape(
                batch_size * seq_len, self.rank, horizon, self.vocab_size
            ),
            ops=points.reshape(batch_size * seq_len, horizon),
        )  # (BT,), (BT, T)
        return p_tilde.reshape(batch_size, seq_len), []  # (B,T)

    def get_norm_consts(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get the normalization constants for the BT distributions.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Norm constants and scale tensors
        """
        batch_size, seq_len, _ = last_hidden_state.size()
        horizon = self._get_horizon(horizon)
        # Get indexed distribution
        params = self._get_params(last_hidden_state, horizon)  # (B, T, R, H, V)
        with profiler.record_function("normalize_cp_tensor"):
            norm_consts = sum_cp_tensor(
                cp_params=params.reshape(
                    batch_size * seq_len, self.rank, horizon, self.vocab_size
                ),
            )
            return (
                norm_consts.reshape(batch_size, seq_len),  # (B, T)
                [],
            )

    def evaluate_at_points_and_get_norm_consts(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        **kwargs,
    ):
        """Evaluate the distribution at the given points and get the normalization constants.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, H)

        Returns:
            tuple:
                - torch.Tensor: Unormalized distribution `p_tilde` at the points of shape (B, T)
                - list: Scale factors for `p_tilde`
                - torch.Tensor: Normalization constants `z` of shape (B, T)
                - list: Scale factors for `z`
        """
        # Get indexed distribution
        batch_size, seq_len, _ = last_hidden_state.size()
        horizon = points.size(-1)
        params = self._get_params(last_hidden_state, horizon)  # (B, T, R, H, V)
        # p_tilde = select_from_cp_tensor(
        #     params.reshape(batch_size * seq_len, self.rank, horizon, self.vocab_size),
        #     points.reshape(batch_size * seq_len, horizon),
        # )
        p_tilde, _ = select_margin_cp_tensor_batched(
            cp_params=params.reshape(
                batch_size * seq_len, self.rank, horizon, self.vocab_size
            ),
            ops=points.reshape(batch_size * seq_len, horizon),
        )  # (BT,), (BT, T)
        norm_consts = sum_cp_tensor(
            cp_params=params.reshape(
                batch_size * seq_len, self.rank, horizon, self.vocab_size
            ),
        )
        return (
            p_tilde.reshape(batch_size, seq_len),
            [],
            norm_consts.reshape(batch_size, seq_len),
            [],
        )
