from abc import abstractmethod
from typing import Callable, Optional, Tuple, List

import torch

from tjdnet.distributions._base import AbstractDist, BaseDistConfig
from tjdnet.distributions._tpnet import TensorParamNet, TensorParamNetConfig


class TJDist(AbstractDist):
    def __init__(self, config: BaseDistConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.horizon = config.horizon
        self.rank = config.rank
        self.param_func = TensorParamNet(
            TensorParamNetConfig(
                in_dim=config.embedding_dim,
                hidden_dim=config.embedding_dim,
                out_dim_decoder=config.vocab_size,
                out_dim_encoder=config.rank * config.horizon,
                positivity_func=config.positivity_func,
            )
        )

    def get_horizon(self, horizon: Optional[int]):
        horizon = self.horizon if horizon is None else horizon
        if horizon > self.horizon:
            raise ValueError(f"Horizon must be less than or equal to {self.horizon}")
        return horizon

    @staticmethod
    def safe_exp(x: torch.Tensor) -> torch.Tensor:
        """Safe exponential function to avoid overflow."""
        return torch.exp(torch.clamp(x, max=20.0))  # Clamp to avoid overflow

    @staticmethod
    def _describe_tensor(tensor: torch.Tensor, name: str):
        """Print essential tensor statistics for debugging."""
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        print(
            f"{name}: {tensor.shape}, min={tensor.min().item():.3f}, max={tensor.max().item():.3f}, "
            f"NaN={nan_count}/{tensor.numel()}, Inf={inf_count}/{tensor.numel()}"
        )

    def _run_diagnostics(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p_tilde: torch.Tensor,
        gammas_p: List[torch.Tensor],
        z_tilde: torch.Tensor,
        gammas_z: List[torch.Tensor],
    ):
        """Run diagnostics to check for NaN values in the tensors."""
        print("=== DIAGNOSTIC REPORT ===")

        # Check inputs first
        self._describe_tensor(x, "input_x")
        self._describe_tensor(y, "input_y")

        # Check outputs from evaluate
        self._describe_tensor(p_tilde, "p_tilde")
        self._describe_tensor(z_tilde, "z_tilde")

        for i, gamma in enumerate(gammas_p):
            self._describe_tensor(gamma, f"gamma_p_{i}")

        for i, gamma in enumerate(gammas_z):
            self._describe_tensor(gamma, f"gamma_z_{i}")

        # Check param_func parameters
        print("--- Parameter Function State ---")
        if self.param_func is not None:
            for name, param in self.param_func.named_parameters():
                self._describe_tensor(param, f"param_func.{name}")

        print("=== END DIAGNOSTIC REPORT ===\n")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Batch size mismatch: z.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}"
            )
        p_tilde, gammas_p, z_tilde, gammas_z = self.evaluate(x, y)

        #  === Health checks >>>
        if torch.isnan(p_tilde).any() or torch.isnan(z_tilde).any():
            print("NaN detected! Running diagnostics...")
            self._run_diagnostics(x, y, p_tilde, gammas_p, z_tilde, gammas_z)

        assert not torch.isnan(p_tilde).any(), "p_tilde NaN"
        assert not torch.isnan(z_tilde).any(), "norm_const NaN"
        if len(gammas_p) == 0 and len(gammas_z) == 0:
            if (p_tilde > z_tilde).any():
                print("p_tilde >= norm_const")
                print("p_tilde:", p_tilde)
                print("norm_const:", z_tilde)

            if not (p_tilde <= z_tilde).all():
                print("p_tilde > norm_const")
            assert (p_tilde <= z_tilde).all(), "p_tilde <= norm_const"
        # <<< Health checks ===

        loss = (
            -torch.log(p_tilde)  # (B, T')
            + torch.log(z_tilde)  # (B, T')
            # Contraction Stability Scale Factors
            - sum([torch.log(z) for z in gammas_p])  # (B, T')
            + sum([torch.log(z) for z in gammas_z])
        )  # (B, T-H)
        return loss

    @abstractmethod
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """Computes P(yh|x, y1:h-1) for h in [1, H].

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
                - Evaluation of the distribution at the points of shape (B, H).
                - Scale tensors (empty list).
                - Normalization constants of shape (B, T).
                - Scale tensors for normalization constants (empty list).
        """
        raise NotImplementedError("evaluate method must be implemented in the subclass")
