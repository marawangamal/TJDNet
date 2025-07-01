from typing import Callable, Optional
import torch

from tjdnet.distributions._base import (
    BaseDistFromLinearConfig,
    BaseDistConfig,
    AbstractDist,
)

from tjdnet.distributions.utils import safe_exp
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched


class CPDist(AbstractDist):
    """CP parameterization of a joint distribution.

    Models the joint distribution p(y1:H | x) as a CP tensor.

    Args:
        config (BaseDistConfig): Configuration for the distribution, including vocab_size, horizon, rank, etc.
        bypass_config (bool, optional): If True, bypasses config validation. Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, config: BaseDistConfig, bypass_config=False, **kwargs):
        super().__init__(config)

        # === config
        self.param_func = None
        self.config = config
        H, R, D, V = (
            self.config.horizon,
            self.config.rank,
            self.config.embedding_dim,
            self.config.vocab_size,
        )
        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
            "safe_exp": safe_exp,
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "leaky_relu": torch.nn.functional.leaky_relu,
            "none": lambda x: x,
        }[config.positivity_func]

        # === params
        self.w_cp = torch.nn.Linear(D, R * H * D, bias=False)
        self.decoder = torch.nn.Parameter(torch.randn(D, V))

    @classmethod
    def from_pretrained(
        cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig, **kwargs
    ):
        raise NotImplementedError("CPDist does not support from_pretrained")

    def get_params(self, x: torch.Tensor, **kwargs):
        B = x.size(0)
        H, R, D, V = (
            self.config.horizon,
            self.config.rank,
            self.config.embedding_dim,
            self.config.vocab_size,
        )
        theta = self.w_cp(x).reshape(B, R, H, D) @ self.decoder  # (B, R, H, V)
        return theta

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
        H = horizon or self.config.horizon
        batch_size = x.size(0)
        dvc = x.device

        # Output tokens will be placed in `y_hat`
        y_hat = torch.empty(batch_size, 0, device=dvc, dtype=torch.long)
        model_head_params = self.get_params(x)  # (B, R, H, V)
        py_tilde_list = []

        # Autoregressive sampling
        # Operations tensor (B, T). Describes batch operations to perform on the CP tensor
        # modelled by `model_head_params`.
        # Example:
        #  y_hat = [[1, 2, 3]]  # (B, T)
        #  ops_tensor = [[1, 2, -2]]  # (B, T)
        #  p_ops_tilde = A^{(1))_1} * A^{(2)}_2 * (𝜮_r A^{(3)}_r)
        for h in range(H):
            ops_tensor = torch.cat(
                (
                    y_hat,  # selection
                    -1  # free leg
                    * torch.ones(batch_size, 1, dtype=torch.long, device=dvc),
                    -2  # marginalization
                    * torch.ones(batch_size, (H - h - 1), dtype=torch.long, device=dvc),
                ),
                dim=1,
            )  # (B, T)
            #  (B, R, H, V) -> (B, V)
            p_ops_tilde, _ = select_margin_cp_tensor_batched(
                cp_params=model_head_params,
                ops=ops_tensor,
            )  # (B, V), (B,) * T
            py_tilde_list.append(p_ops_tilde)
            next_token = sample_fn(p_ops_tilde).unsqueeze(1)  # (B,1)

            y_hat = torch.cat([y_hat, next_token], dim=1)
        py_tilde = torch.stack(py_tilde_list, dim=1)  # (B, H, V)
        if return_logits:  # don't normalize
            return y_hat, py_tilde
        return y_hat, py_tilde / py_tilde.sum(dim=-1, keepdim=True)  # (B, H)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ):
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Batch size mismatch: z.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}"
            )
        # Get indexed distribution
        H, R, V = (
            self.config.horizon,
            self.config.rank,
            self.config.vocab_size,
        )
        B = x.size(0)
        params = self.get_params(x)  # (B, R, H, V)
        p_tilde, gammas_p = select_margin_cp_tensor_batched(
            cp_params=params.reshape(B, R, H, V),
            ops=y.reshape(B, H),
        )  # (B,), (B, H)
        z_tilde, gammas_z = select_margin_cp_tensor_batched(
            cp_params=params.reshape(B, R, H, V),
            ops=torch.full(
                (B, H),
                -2,
                dtype=torch.long,
                device=x.device,
            ),
        )
        loss = (
            -torch.log(p_tilde)  # (B, T')
            + torch.log(z_tilde)  # (B, T')
            # Contraction Stability Scale Factors
            - sum([torch.log(z) for z in gammas_p])  # (B, T')
            + sum([torch.log(z) for z in gammas_z])
        )  # (B, T-H)
        return loss

    # @classmethod
    # def from_pretrained(
    #     cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig, **kwargs
    # ):
    #     """Create a CP distribution from a linear layer.

    #     Args:
    #         linear (torch.nn.Linear): Linear layer to use as a base. Shape: (D, V)
    #         config (BaseDistFromLinearConfig): Configuration for the distribution.

    #     Returns:
    #         CPDist: CP distribution with the given configuration.
    #     """

    #     vocab_size, hidden_dim = linear.weight.shape
    #     use_bias_decoder = False
    #     if linear.bias is not None:
    #         use_bias_decoder = True
    #         raise Warning("CPDist: Skiping bias initialization.")

    #     param_net_conf = config.param_net.to_dict()
    #     param_net_conf["hidden_dim"] = hidden_dim
    #     param_net_conf["out_dim_decoder"] = vocab_size
    #     param_net_conf["use_bias_decoder"] = use_bias_decoder

    #     obj = cls(
    #         config=BaseDistConfig(
    #             vocab_size=vocab_size,
    #             horizon=config.horizon,
    #             rank=config.rank,
    #             param_net=TensorParamNetConfig(**param_net_conf),
    #         ),
    #         **kwargs,
    #     )

    #     # Initialize the parameters in obj.tensor_param_net
    #     # with the parameters from the linear layer
    #     obj.param_func.decoder.weight.data = linear.weight.data  # type: ignore
    #     if use_bias_decoder:
    #         obj.param_func.decoder.bias.data = linear.bias.data  # type: ignore
    #     return obj
