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
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        lr: float = 1e-2,
        epochs: int = 1000,
        batch_size: int = 512,
        rtol: float = 1e-3,  # relative tolerance (fractional improvement)
        patience: int = 10,
    ):
        """Train CPRegressor with Adam.

        Args:
            X (torch.Tensor): Training data inputs. Shape: (B, H).
            y (torch.Tensor): Training data targets. Shape: (B,).
            lr (float, optional): Learning rate. Defaults to 1e-2.
            epochs (int, optional): Max num epochs. Defaults to 1000.
            batch_size (int, optional): Batch Size. Defaults to 512.
            rtol (float, optional): Relative tolerance for early stopping. Defaults to 1e-3.
            patience (int, optional): Early stopping patience. Defaults to 10.

        Returns:
            CPRegressor: Fitted model.
        """
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_loss, wait = float("inf"), 0
        epoch_bar = tqdm(range(epochs), desc="epochs", leave=False)

        for _ in epoch_bar:
            running = 0.0
            self.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(self(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)

            epoch_loss = running / len(ds)
            epoch_bar.set_postfix(loss=f"{epoch_loss:.6f}")

            # ------------- early-stopping with relative tolerance -------------
            if best_loss == float("inf"):
                best_loss = epoch_loss
                continue

            improvement = (best_loss - epoch_loss) / best_loss
            if improvement > rtol:
                best_loss, wait = epoch_loss, 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return self
