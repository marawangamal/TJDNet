from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class CPRegressor(nn.Module):
    """
    CP regression with rank-R factors learned by Adam.

        ŷ = Σ_r w_r · ∏_{m=1..H} A_m[i_m,r] + b
    """

    def __init__(self, vocab_size: int, horizon: int, rank: int, device: str = "cpu"):
        super().__init__()
        self.V, self.H, self.R = vocab_size, horizon, rank
        self.to(device)

        self.factors = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.V, self.R, device=device))
                for _ in range(self.H)
            ]
        )
        self.weights = nn.Parameter(torch.ones(self.R, device=device))
        self.bias = nn.Parameter(torch.zeros(1, device=device))

    # ------------------------------------------------------------------ #
    # forward / predict
    # ------------------------------------------------------------------ #
    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # coords: (N, H)
        prod = torch.ones(coords.size(0), self.R, device=coords.device)
        for m in range(self.H):
            prod *= self.factors[m][coords[:, m]]
        return (prod * self.weights).sum(1) + self.bias

    def predict(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self(coords)

    # ------------------------------------------------------------------ #
    # fit with tqdm + simple early stopping
    # ------------------------------------------------------------------ #

    def _eval_loss(self, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
        with torch.no_grad():
            mse_loss = nn.MSELoss()
            pred = self.predict(x_val)
            return mse_loss(pred, y_val).item()

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        *,
        lr: float = 1e-4,
        epochs: int = 1000,
        batch_size: int = 512,
        rtol: float = 5e-2,  # relative tolerance
        atol: float = 1e-2,  # absolute tolerance
        patience: int = 10,
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
        mse = nn.MSELoss()

        best_loss, wait = float("inf"), 0
        bar = tqdm(range(epochs), desc="epochs", leave=False)

        improvement_checks = [
            # relative
            lambda best, curr: (best - curr) / best > rtol,
            # absolute
            lambda best, curr: best - curr > atol,
        ]

        for _ in bar:
            running = 0.0
            self.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = mse(self(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)

            train_loss = running / len(ds)
            eval_loss = (
                self._eval_loss(x_val, y_val)
                if x_val is not None and y_val is not None
                else None
            )

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
            else:
                wait += 1
                if wait >= patience:
                    break

        return self
