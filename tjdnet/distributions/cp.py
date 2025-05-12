from git import Optional
import torch

from tjdnet.distributions._tjdist import (
    BaseDistConfig,
    BaseDistFromLinearConfig,
    TJDist,
)

from tjdnet.tensorops.cp import select_margin_cp_tensor_batched
from tjdnet.utils import sample_topk


class CPDist(TJDist):
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

    @classmethod
    def from_linear(
        cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig, **kwargs
    ):
        """Create a CP distribution from a linear layer.

        Args:
            linear (torch.nn.Linear): Linear layer to use as a base. Shape: (D, V)
            config (BaseDistFromLinearConfig): Configuration for the distribution.

        Returns:
            CPDist: CP distribution with the given configuration.
        """

        n_emb, vocab_size = linear.weight.shape
        if linear.bias is not None:
            raise Warning("CPDist: Skiping bias initialization.")

        obj = cls(
            config=BaseDistConfig(
                vocab_size=vocab_size,
                horizon=config.horizon,
                rank=config.rank,
                param_net=config.param_net,
            ),
            **kwargs,
        )

        # Initialize the parameters in obj.tensor_param_net
        # with the parameters from the linear layer
        obj.param_func.linear.weight.data = linear.weight.data
        return obj

    def get_params(self, x: torch.Tensor, **kwargs):
        B = x.size(0)
        params = self.param_func(x)  # (B, R * H, V)
        params_reshaped = params.reshape(B, self.rank, self.horizon, self.vocab_size)
        return params_reshaped  # (B, R, H, V)  // H* is model level horizon

    def sample(
        self,
        x: torch.Tensor,
        horizon: Optional[int] = None,
        do_sample: bool = False,
        top_k: int = 200,
        **kwargs,
    ):
        horizon = self.get_horizon(horizon)
        batch_size = x.size(0)
        dvc = x.device
        y_hat = torch.empty(batch_size, 0, device=dvc, dtype=torch.long)
        model_head_params = self.get_params(x)  # B, R, H, V)
        py_tilde_list = []
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
            #  (B, R, H, V) -> (B, V)
            p_ops_tilde, _ = select_margin_cp_tensor_batched(
                cp_params=model_head_params,
                ops=ops_tensor,
            )  # (B, V), (B,) * T
            py_tilde_list.append(p_ops_tilde)
            if do_sample:
                next_token = sample_topk(p_ops_tilde, top_k, num_samples=1)
            else:  # greedy sampling
                next_token = sample_topk(p_ops_tilde, 1, num_samples=1)
            y_hat = torch.cat([y_hat, next_token], dim=1)
        py_tilde = torch.stack(py_tilde_list, dim=1)  # (B, H, V)
        return y_hat, py_tilde / py_tilde.sum(dim=-1, keepdim=True)  # (B, H)

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ):
        # Get indexed distribution
        horizon = self.horizon
        B = x.size(0)
        params = self.get_params(x)  # (B, T, R, H, V)
        p_tilde, p_tilde_scale_factors = select_margin_cp_tensor_batched(
            cp_params=params.reshape(B, self.rank, horizon, self.vocab_size),
            ops=y.reshape(B, horizon),
        )  # (B,), (B, H)
        norm_consts, norm_consts_scale_factors = select_margin_cp_tensor_batched(
            cp_params=params.reshape(B, self.rank, horizon, self.vocab_size),
            ops=torch.full(
                (B, horizon),
                -2,
                dtype=torch.long,
                device=x.device,
            ),
        )
        return (p_tilde, p_tilde_scale_factors, norm_consts, norm_consts_scale_factors)
