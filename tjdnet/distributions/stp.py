from typing import Callable, Optional
import torch

from tjdnet.distributions._base import (
    AbstractDist,
    BaseDistConfig,
    BaseDistFromLinearConfig,
)


class STPDist(AbstractDist):
    def __init__(
        self,
        config: BaseDistConfig,
        **kwargs,
    ):
        """Basic 1D entropy distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        super().__init__()
        self.vocab_size = config.vocab_size
        self.rank = config.rank
        self.horizon = config.horizon
        self.decoder = torch.nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

    @classmethod
    def from_pretrained(
        cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig, **kwargs
    ):
        raise NotImplementedError("CPDist does not support from_pretrained")

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
    #     # ==== Assertions ====
    #     rules = [
    #         {
    #             "fn": lambda config: config.rank == 1,
    #             "msg": "Rank must be 1 for base distribution",
    #         },
    #         {
    #             "fn": lambda config: config.horizon == 1,
    #             "msg": "Horizon must be 1 for base distribution",
    #         },
    #     ]
    #     for rule in rules:
    #         if not rule["fn"](config):
    #             raise ValueError(rule["msg"])
    #     # ====================

    #     vocab_size, hidden_dim = linear.weight.shape
    #     use_bias_encoder = False
    #     if linear.bias is not None:
    #         use_bias_encoder = True
    #         raise Warning("BaseDist: Skiping bias initialization.")

    #     param_net_conf = config.param_net.to_dict()
    #     param_net_conf["hidden_dim"] = hidden_dim
    #     param_net_conf["use_decoder"] = False

    #     obj = cls(
    #         config=BaseDistConfig(
    #             vocab_size=vocab_size,
    #             horizon=config.horizon,
    #             rank=config.rank,
    #             param_net=config.param_net,
    #         ),
    #         **kwargs,
    #     )

    #     # Initialize the parameters in obj.tensor_param_net
    #     # with the parameters from the linear layer
    #     obj.param_func.w.weight.data = linear.weight.data  # type: ignore
    #     if use_bias_encoder:
    #         obj.param_func.w.bias.data = linear.bias.data  # type: ignore
    #     return obj

    def sample(
        self,
        x: torch.Tensor,  # (B, T, D)
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        **kwargs,
    ):
        if horizon and horizon > 1:
            raise ValueError("Horizon must be 1 for base distribution")
        model_head_params = self.decoder(x)  # (B, V)
        y_hat = sample_fn(model_head_params).unsqueeze(1)  # (B, 1)
        py = model_head_params.unsqueeze(1)  # (B, 1, V)
        if return_logits:
            return y_hat, py
        return y_hat, py / py.sum(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes loss for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
        """
        # preds = torch.nn.functional.softmax(self.decoder(x), dim=-1)  # (B, V)
        # p_tilde_select = preds.gather(dim=-1, index=y).squeeze(-1)  # (B,)
        # return -(p_tilde_select.log())

        # CE loss (should be equivalent)
        preds = self.decoder(x)  # (B, V)
        loss = torch.nn.functional.cross_entropy(
            preds, y.squeeze(-1), reduction="none"
        )  # (B,)
        return loss
