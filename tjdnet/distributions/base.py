from typing import List, Optional, Tuple
import torch
import line_profiler

from tjdnet.distributions._base import BaseDistribution, BaseDistConfig
from tjdnet.tensorops.common import sample_from_tensor_dist
from tjdnet.utils import sample_topk


class BaseDist(BaseDistribution):
    def __init__(
        # self,
        # n_embd: int,
        # vocab_size,
        # positivity_func: str = "exp",
        # horizon: int = 1,
        # **kwargs,
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
        assert config.horizon == 1, "Only horizon=1 is supported for now"
        config.param_net.use_decoder = False  # Set TPNet decoder to False
        config.param_net.out_dim_encoder = config.vocab_size
        config.param_net.hidden_dim = 1
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.rank = config.rank
        self.horizon = config.horizon

    def init_params(
        self, pt_weight: torch.Tensor, pt_bias: Optional[torch.Tensor]
    ) -> None:
        """
        Initialize the weights (and optionally the bias) of a single-layer
        TensorParamNet with a given parameter tensor.

        Args:
            params (torch.Tensor): A tensor of shape matching the linear layer's
                weight shape, i.e. (out_dim, in_dim).
        """
        # Identify the linear layer (it's the last item in self.param_func.network)
        linear_layer = self.param_func.network[-1]
        if not isinstance(linear_layer, torch.nn.Linear):
            raise ValueError(
                f"Expected the last layer to be nn.Linear, but got {type(linear_layer)}"
            )

        # Shape check
        expected_shape = linear_layer.weight.shape  # (out_dim, in_dim)
        if pt_weight.shape != expected_shape:
            raise ValueError(
                f"Expected params of shape {expected_shape}, but got {pt_weight.shape}"
            )

        if pt_bias is not None and pt_bias.shape != linear_layer.bias.shape:
            raise ValueError(
                f"Expected bias of shape {linear_layer.bias.shape}, but got {pt_bias.shape}"
            )

        # Copy parameters
        with torch.no_grad():
            linear_layer.weight.copy_(pt_weight)
            if pt_bias is not None:
                linear_layer.bias.copy_(pt_bias)
            else:
                linear_layer.bias.zero_()

    def _get_params(self, last_hidden_state: torch.Tensor, **kwargs) -> torch.Tensor:
        p_tilde = self.param_func(last_hidden_state)  # (B, T, 1, V)
        return p_tilde.squeeze(-1)  # (B, T, V)

    def get_dist(
        self,
        hidden_state: torch.Tensor,
        ops: torch.Tensor,
        use_cache: bool,
        save_cache: bool,
        **kwargs,
    ):
        """Get distribution specified by ops."""
        assert ops.size(-1) == 1, "Only 1D points are supported"
        p_tilde = self._get_params(hidden_state)  # (B, 1, V)
        p_tilde = p_tilde.reshape(-1)  # (B, V)
        return p_tilde, []

    def sample(
        self,
        hidden_state: torch.Tensor,  # (B, T, D)
        horizon: Optional[int] = None,
        do_sample: bool = False,
        top_k: int = 200,
        **kwargs,
    ):
        if horizon and horizon > 1:
            raise ValueError("Horizon must be 1 for base distribution")
        model_head_params = self._get_params(hidden_state[:, -1:, :])  # (B, 1, V)
        p_tilde = sample_topk(
            model_head_params.squeeze(1), top_k=top_k if do_sample else 1  # topk/greedy
        )
        return p_tilde

    def evaluate_at_points(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, 1)
            is_normalized (bool, optional): Whether the points are normalized. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points of Shape (B, 1) and scale_tensors (empty list)
        """
        # Get indexed distribution
        assert points.size(-1) == 1, "Only 1D points are supported"
        p_tilde = self._get_params(last_hidden_state)  # (B, T, V)
        batch_size, seq_len, _ = last_hidden_state.size()

        # (B, T, R*H*V) => (B, T)
        p_tilde_select = torch.gather(
            p_tilde.reshape(batch_size * seq_len, self.vocab_size),
            dim=1,
            index=points.reshape(batch_size * seq_len, self.horizon),
        )  # (B*T, 1)
        return p_tilde_select.reshape(batch_size, seq_len), []  # (B*T)

    def get_norm_consts(
        self, last_hidden_state: torch.Tensor, horizon: int, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get the normalization constants for the BT distributions.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Norm constants and scale tensors
        """
        # Get indexed distribution
        batch_size, seq_len, _ = last_hidden_state.size()
        p_tilde = self.positivity_func(self.param_func(last_hidden_state))  # (B, T, V)
        norm_consts = torch.sum(p_tilde, dim=-1).reshape(-1)  # (B*T)
        return (
            norm_consts.reshape(batch_size, seq_len),
            [],
        )

    def evaluate_at_points_and_get_norm_consts(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        **kwargs,
    ):
        """Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, 1)
            is_normalized (bool, optional): Whether the points are normalized. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: Unormalized distribution `p_tilde` at the points of shape (B, T)
                - list: Scale factors for `p_tilde`
                - torch.Tensor: Normalization constants `z` of shape (B, T)
                - list: Scale factors for `z`
        """
        # Get indexed distribution
        assert points.size(-1) == 1, "Only 1D points are supported"
        p_tilde = self._get_params(last_hidden_state)  # (B, T, V)
        batch_size, seq_len, _ = last_hidden_state.size()

        # (B, T, R*H*V) => (B, T)
        p_tilde_select = torch.gather(
            p_tilde.reshape(batch_size * seq_len, self.vocab_size),
            dim=1,
            index=points.reshape(batch_size * seq_len, self.horizon),
        )  # (B*T, 1)
        norm_consts = torch.sum(p_tilde, dim=-1).reshape(-1)  # (B*T)
        return (
            p_tilde_select.reshape(batch_size, seq_len),
            [],
            norm_consts.reshape(batch_size, seq_len),
            [],
        )
