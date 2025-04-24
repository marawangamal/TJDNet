from copy import deepcopy
from typing import Literal, Optional, Union
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def mean_squared_relative_error(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Element-wise relative error |y_pred - y_true| / |y_true|.

    Args:
        y_pred: Predicted values.
        y_true: True (reference) values.
        eps: Small constant added to the denominator to prevent division by zero.

    Returns:
        torch.Tensor: Relative error tensor.
    """
    # diff = torch.abs(y_pred - y_true)
    # denom = torch.abs(y_true).clamp_min(eps)  # ensures ≥ eps
    # return (diff / denom).mean()
    return torch.norm(y_pred - y_true) / torch.clamp_min(torch.norm(y_true), eps)


def mean_absolute_relative_error(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-12
):
    return (torch.abs(y_pred - y_true) / y_true.abs().clamp_min(eps)).mean()


class CPRegressor(nn.Module):
    """CP regression with rank-R factors learned by Adam.

    The model predicts a scalar ŷ from an H-dimensional multi-index i = (i₁,…,i_H) by

    .. math::
        \\hat{y} = \\sum_{r=1}^R w_r \\prod_{m=1}^H  A_m[i_m,r]

    """

    def __init__(
        self,
        vocab_size: int,
        horizon: int,
        rank: int,
        device: str = "cpu",
        init_method: str = "normal",
        verbose: bool = False,
    ):
        super().__init__()
        self.V, self.H, self.R = vocab_size, horizon, rank
        self.verbose = verbose
        self.to(device)

        init_fn = {
            # NOTE: scale by rank
            "normal": lambda: torch.randn(self.V, self.R, device=device) / self.R,
            "uniform": lambda: torch.rand(self.V, self.R, device=device),
            "xavier": lambda: nn.init.xavier_normal_(
                torch.empty(self.V, self.R, device=device)
            ),
            "kaiming": lambda: nn.init.kaiming_normal_(
                torch.empty(self.V, self.R, device=device)
            ),
            "zeros": lambda: torch.zeros(self.V, self.R, device=device),
        }

        self.factors = nn.ParameterList(
            [
                nn.Parameter(init_fn[init_method](), requires_grad=True)
                for _ in range(self.H)
            ]
        )
        self.weights = nn.Parameter(torch.ones(self.R, device=device))

    # ------------------------------------------------------------------ #
    # forward / predict
    # ------------------------------------------------------------------ #
    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # coords: (N, H)
        prod = torch.ones(coords.size(0), self.R, device=coords.device)
        for m in range(self.H):
            prod *= self.factors[m][coords[:, m]]
        return (prod * self.weights).sum(1)

    def predict(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self(coords)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # fit with tqdm + simple early stopping
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        *,
        lr: float = 1e-3,
        epochs: int = 5000,
        min_epochs: int = 500,
        batch_size: int = 512,
        rtol: float = 1e-3,  # >= 0.1% relative tolerance
        atol: float = 1e-5,  # absolute tolerance
        patience: int = 10,
        loss_type: Literal["msre", "mare", "mse"] = "msre",
        **kwargs,
    ):
        """Train CPRegressor with Adam.

        Args:
            X (torch.Tensor): Training data inputs. Shape: (B, H).
            y (torch.Tensor): Training data targets. Shape: (B,).
            lr (float, optional): Learning rate. Defaults to 1e-2.
            epochs (int, optional): Max num epochs. Defaults to 1000.
            batch_size (int, optional): Batch Size. Defaults to 512.
            rtol (float, optional): Relative tolerance for early stopping. Defaults to 5e-2 (i.e 5%).
            patience (int, optional): Early stopping patience. Defaults to 10.

        Returns:
            CPRegressor: Fitted model.
        """
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = {
            "msre": mean_squared_relative_error,
            "mare": mean_absolute_relative_error,
            "mse": nn.MSELoss(),
        }[loss_type]

        best_loss, wait = float("inf"), 0
        best_state = None
        bar = tqdm(range(epochs), desc="epochs", leave=False)

        improvement_checks = [
            # relative
            lambda best, curr: (best - curr) / best > rtol,
            # absolute
            lambda best, curr: best - curr > atol,
        ]

        for epoch in bar:
            running = 0.0
            self.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(self(xb), yb) * 1
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)

            train_loss = running / len(ds)

            # Compute validation loss if provided
            eval_loss = None
            if x_val is not None and y_val is not None:
                with torch.no_grad():
                    pred = self.predict(x_val)
                    eval_loss = loss_fn(pred, y_val).item()

            bar.set_postfix(
                train_loss=f"{train_loss:.6f}",
                eval_loss=f"{eval_loss:.6f}" if eval_loss else None,
            )

            epoch_loss = train_loss if eval_loss is None else eval_loss

            # initialise best_loss after first epoch
            if best_loss == float("inf"):
                best_loss = epoch_loss
                continue

            if any(fn(best_loss, epoch_loss) for fn in improvement_checks):
                best_loss, wait = epoch_loss, 0  # significant improvement
                best_state = deepcopy(self.state_dict())
            else:
                # Check if minimum epochs reached
                if epoch < min_epochs:
                    continue

                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            self.print(
                f"Early stopping at epoch {epoch} with best loss {best_loss:.6f}"
            )
            self.load_state_dict(best_state)

        return self
