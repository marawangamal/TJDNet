from git import Optional
import torch

from tjdnet.distributions._base import BaseDistConfig, BaseDistribution
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched
from tjdnet.utils import sample_topk


# TODO: maybe we can simplify this with einsum and sparse tensors
class CPDist(BaseDistribution):
    def __init__(self, config: BaseDistConfig, **kwargs):
        """CP Distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        # config.param_net.out_dim_encoder = config.rank * config.horizon * config.vocab_size
        config.param_net.out_dim_encoder = config.rank * config.horizon
        config.param_net.out_dim_decoder = config.vocab_size
        super().__init__(config)

    def _get_params(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ):
        batch_size, seq_len, _ = last_hidden_state.size()
        params = self.param_func(last_hidden_state)  # (B, T, R * H, V)
        params_reshaped = params.reshape(
            batch_size, seq_len, self.rank, self.horizon, self.vocab_size
        )
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
    ):
        horizon = self._get_horizon(horizon)
        batch_size = hidden_state.size(0)
        dvc = hidden_state.device
        y_hat = torch.empty(batch_size, 0, device=dvc, dtype=torch.long)
        model_head_params = self._get_params(hidden_state[:, -1:, :]).squeeze(
            1
        )  # (B, 1, R, H*, V) => (B, R, H*, V)
        py_list = []
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
            py_list.append(p_ops_tilde)
            if do_sample:
                next_token = sample_topk(p_ops_tilde, top_k, num_samples=1)
            else:  # greedy sampling
                next_token = sample_topk(p_ops_tilde, 1, num_samples=1)
            y_hat = torch.cat([y_hat, next_token], dim=1)
        py = torch.stack(py_list, dim=1)  # (B, H, V)
        return y_hat, py  # (B, H)

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
        p_tilde, p_tilde_scale_factors = select_margin_cp_tensor_batched(
            cp_params=params.reshape(
                batch_size * seq_len, self.rank, horizon, self.vocab_size
            ),
            ops=points.reshape(batch_size * seq_len, horizon),
        )  # (BT,), (BT, T)
        norm_consts, norm_consts_scale_factors = select_margin_cp_tensor_batched(
            cp_params=params.reshape(
                batch_size * seq_len, self.rank, horizon, self.vocab_size
            ),
            ops=torch.full(
                (batch_size * seq_len, horizon),
                -2,
                dtype=torch.long,
                device=last_hidden_state.device,
            ),
        )
        return (
            p_tilde.reshape(batch_size, seq_len),
            [s.reshape(batch_size, seq_len) for s in p_tilde_scale_factors],
            norm_consts.reshape(batch_size, seq_len),
            [s.reshape(batch_size, seq_len) for s in norm_consts_scale_factors],
        )
