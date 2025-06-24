from typing import Callable
from git import Optional
import torch

from tjdnet.distributions._tjdist import (
    BaseDistConfig,
    BaseDistFromLinearConfig,
    TJDist,
)

from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.tensorops.cp import (
    select_margin_cp_tensor_batched,
    select_margin_cp_tensor_batched_w_decoder,
)


def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Safe exponential function to avoid overflow."""
    return torch.exp(torch.clamp(x, max=20.0))  # Clamp to


class CPEffDist(TJDist):
    def __init__(self, config: BaseDistConfig, **kwargs):
        """CP Distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        # config.param_net.out_dim_encoder = config.rank * config.horizon
        # config.param_net.out_dim_decoder = config.vocab_size

        # Note: hacky way to ignore the param_net
        config.param_net.out_dim_encoder = 1
        config.param_net.out_dim_decoder = 1
        super().__init__(config)
        self.param_func = None

        self.cp_hidden_dim = config.param_net.hidden_dim
        self.cp_rank = config.rank
        self.cp_horizon = config.horizon
        self.cp_vocab_size = config.vocab_size

        # CP projection
        self.cp_w = torch.nn.Linear(
            config.param_net.in_dim,
            config.rank * config.horizon * config.param_net.hidden_dim,
        )
        self.cp_decoder = torch.nn.Parameter(
            torch.randn(config.param_net.hidden_dim, config.vocab_size)
        )

    @classmethod
    def from_pretrained(
        cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig, **kwargs
    ):
        """Create a CP distribution from a linear layer.

        Args:
            linear (torch.nn.Linear): Linear layer to use as a base. Shape: (D, V)
            config (BaseDistFromLinearConfig): Configuration for the distribution.

        Returns:
            CPDist: CP distribution with the given configuration.
        """

        vocab_size, hidden_dim = linear.weight.shape
        use_bias_decoder = False
        if linear.bias is not None:
            use_bias_decoder = True
            raise Warning("CPDist: Skiping bias initialization.")

        param_net_conf = config.param_net.to_dict()
        param_net_conf["hidden_dim"] = hidden_dim
        param_net_conf["out_dim_decoder"] = vocab_size
        param_net_conf["use_bias_decoder"] = use_bias_decoder

        obj = cls(
            config=BaseDistConfig(
                vocab_size=vocab_size,
                horizon=config.horizon,
                rank=config.rank,
                param_net=TensorParamNetConfig(**param_net_conf),
            ),
            **kwargs,
        )

        # Initialize the parameters in obj.tensor_param_net
        # with the parameters from the linear layer
        obj.param_func.decoder.weight.data = linear.weight.data  # type: ignore
        if use_bias_decoder:
            obj.param_func.decoder.bias.data = linear.bias.data  # type: ignore
        return obj

    def get_params(self, x: torch.Tensor, **kwargs):
        B = x.size(0)
        params = safe_exp(self.cp_w(x))  # (B, RHd)
        params_reshaped = params.reshape(
            B, self.cp_rank, self.cp_horizon, self.cp_hidden_dim
        )
        return params_reshaped  # (B, R, H, d)  // H* is model level horizon

    def sample(
        self,
        x: torch.Tensor,
        # (B, D) -> (B,)
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        refine: bool = False,
        **kwargs,
    ):
        """Computes P(yh|x, y1:h-1) for h in [1, H].

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            sample_fn (Callable): Sampling function.
            horizon (Optional[int]): Horizon for sampling. Must be <= self.horizon.
            return_logits (bool): Whether to return logits or probabilities.
            refine (bool): Whether to refine the sampling process.

        Returns:
            tuple:
                - Evaluation of the distribution at the points of shape (B, H).
                - Probabilities of shape (B, H, V) or logits of shape (B, H, V).
        """
        horizon = self.get_horizon(horizon)  # Possibly override model horizon
        batch_size = x.size(0)
        dvc = x.device

        # Output tokens will be placed in `y_hat`
        y_hat = torch.empty(batch_size, 0, device=dvc, dtype=torch.long)
        model_head_params = self.get_params(x)  # (B, R, H, d)
        py_tilde_list = []

        # Autoregressive sampling
        # Operations tensor (B, T). Describes batch operations to perform on the CP tensor
        # modelled by `model_head_params`.
        # Example:
        #  y_hat = [[1, 2, 3]]  # (B, T)
        #  ops_tensor = [[1, 2, -2]]  # (B, T)
        #  p_ops_tilde = A^{(1))_1} * A^{(2)}_2 * (𝜮_r A^{(3)}_r)
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
            p_ops_tilde, _ = select_margin_cp_tensor_batched_w_decoder(
                cp_params=model_head_params,
                ops=ops_tensor,
                decoder=self.cp_decoder,
            )  # (B, V), (B,) * T
            py_tilde_list.append(p_ops_tilde)
            next_token = sample_fn(p_ops_tilde).unsqueeze(1)  # (B,1)

            y_hat = torch.cat([y_hat, next_token], dim=1)
        py_tilde = torch.stack(py_tilde_list, dim=1)  # (B, H, V)
        if return_logits:  # don't normalize
            return y_hat, py_tilde
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
        params = self.get_params(x)  # (B, R, H, d)
        p_tilde, p_tilde_scale_factors = select_margin_cp_tensor_batched_w_decoder(
            cp_params=params,
            ops=y.reshape(B, horizon),
            decoder=self.cp_decoder,
        )  # (B,), (B, H)
        norm_consts, norm_consts_scale_factors = (
            select_margin_cp_tensor_batched_w_decoder(
                cp_params=params.reshape(B, self.rank, horizon, self.vocab_size),
                ops=torch.full(
                    (B, horizon),
                    -2,
                    dtype=torch.long,
                    device=x.device,
                ),
                decoder=self.cp_decoder,
            )
        )
        return (p_tilde, p_tilde_scale_factors, norm_consts, norm_consts_scale_factors)
