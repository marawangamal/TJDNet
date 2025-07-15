from typing import Callable, Optional
import torch

from tjdnet.distributions._base import BaseDistFromLinearConfig
from tjdnet.distributions._tjdist import BaseDistConfig, TJDist

from tjdnet.distributions._tpnet import safe_exp
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched


class CPDropDist(TJDist):
    def __init__(self, config: BaseDistConfig, bypass_config=False, **kwargs):
        super().__init__(config)
        self.param_func = None
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
        self.config = config

        # Encoder
        self.w_encoder = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(
                torch.empty(
                    config.embedding_dim,
                    config.embedding_dim * config.rank * config.horizon,
                )
            )
        )
        self.b_encoder = torch.nn.Parameter(
            torch.zeros(config.embedding_dim * config.rank * config.horizon)
        )

        # Decoder
        self.w_decoder = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(config.embedding_dim, config.vocab_size))
        )
        self.b_decoder = torch.nn.Parameter(torch.zeros(config.vocab_size))

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

    @classmethod
    def from_pretrained(
        cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig, **kwargs
    ):
        raise NotImplementedError("CPDist does not support from_pretrained")

    def get_params(self, x: torch.Tensor, **kwargs):
        # Setup
        D, R, H = self.config.embedding_dim, self.config.rank, self.config.horizon

        # Encoder: (B, D) -> (B, D, R', H)
        mask_keep = (
            torch.rand(R, device=x.device) > self.config.dropout
            if not self.training
            else torch.ones(R, device=x.device) == torch.tensor(1.0)
        )
        # If all ranks are dropped, keep all ranks
        mask_keep = (
            torch.ones(R, device=x.device) == torch.tensor(1.0)
            if mask_keep.sum() == 0
            else mask_keep
        )

        gamma = 1 / (1 - self.config.dropout) if self.training else 1.0
        w_encoder = self.w_encoder.reshape(D, D, R, H)[:, :, mask_keep]  # (D, D, R', H)
        z = gamma * (
            torch.einsum("be,edrh->bdrh", x, w_encoder)
            + self.b_encoder.reshape(D, R, H)[:, mask_keep]
        )  # (B, D, R', H)

        # Decoder: (B, D, R', H) -> (B, R', H, V)
        theta = torch.einsum("bdrh,dv->brhv", z, self.w_decoder) + self.b_decoder

        # Positivity
        theta = self.positivity_func(theta)

        return theta  # (B, R', H, V)

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
        dvc = x.device

        # Output tokens will be placed in `y_hat`
        model_head_params = self.get_params(x)  # (B, R, H, V)
        B, R, H, V = model_head_params.shape
        y_hat = torch.empty(B, 0, device=dvc, dtype=torch.long)

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
                    -1 * torch.ones(B, 1, dtype=torch.long, device=dvc),  # free leg
                    -2  # marginalization
                    * torch.ones(B, (H - h - 1), dtype=torch.long, device=dvc),
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

    def evaluate(
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
        params = self.get_params(x)  # (B, R, H, V)
        B, R, H, V = params.shape
        p_tilde, p_tilde_scale_factors = select_margin_cp_tensor_batched(
            cp_params=params.reshape(B, R, H, V),
            ops=y.reshape(B, H),
        )  # (B,), (B, H)
        norm_consts, norm_consts_scale_factors = select_margin_cp_tensor_batched(
            cp_params=params.reshape(B, R, H, V),
            ops=torch.full(
                (B, H),
                -2,
                dtype=torch.long,
                device=x.device,
            ),
        )
        return (p_tilde, p_tilde_scale_factors, norm_consts, norm_consts_scale_factors)
