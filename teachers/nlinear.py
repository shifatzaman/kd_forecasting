import torch
import torch.nn as nn
from teachers.base import TeacherModel
from core.dataset import make_windows


class NLinear(nn.Module):
    """
    NLinear: Normalization-Linear model for time series forecasting.

    Subtracts the last value before forecasting, then adds it back.
    This helps the model focus on changes rather than absolute values.

    Input:  [B, lookback, 1]
    Output: [B, horizon]
    """

    def __init__(self, lookback: int, horizon: int):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, x):
        """
        x: [B, lookback, 1]
        """
        # Remove channel dimension
        x = x.squeeze(-1)  # [B, lookback]

        # Subtract last value (normalize)
        last_val = x[:, -1:].clone()  # [B, 1]
        x_normalized = x - last_val

        # Linear forecast
        out = self.linear(x_normalized)  # [B, horizon]

        # Add back the last value
        return out + last_val


class NLinearTeacher(TeacherModel):
    """
    NLinear teacher wrapper following TeacherModel interface.
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        epochs: int,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NLinear(
            lookback=lookback,
            horizon=horizon,
        ).to(self.device)

    def fit(self, series):
        """
        Train NLinear on a single univariate series.
        """
        X, Y = make_windows(
            series,
            lookback=self.lookback,
            horizon=self.horizon,
        )

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()

        self.model.train()
        for _ in range(self.epochs):
            preds = self.model(X)
            loss = loss_fn(preds, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict(self, X):
        """
        X: numpy array [N, lookback]
        Returns: numpy array [N, horizon]
        """
        self.model.eval()

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        preds = self.model(X)

        return preds.cpu().numpy()
