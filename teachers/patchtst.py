import torch
import torch.nn as nn
from teachers.base import TeacherModel
from core.dataset import make_windows


class PatchTST(nn.Module):
    """
    Lightweight PatchTST implementation for univariate time series.

    Input:  [B, lookback, 1]
    Output: [B, horizon]
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        patch_len: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert lookback % patch_len == 0, (
            f"lookback ({lookback}) must be divisible by patch_len ({patch_len})"
        )

        self.lookback = lookback
        self.horizon = horizon
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Forecast head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * self.n_patches),
            nn.Linear(d_model * self.n_patches, horizon),
        )

    def forward(self, x):
        """
        x: [B, lookback, 1]
        """
        # Remove channel dim
        x = x.squeeze(-1)  # [B, lookback]

        # Create patches
        patches = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.patch_len
        )  # [B, n_patches, patch_len]

        # Embed patches
        z = self.patch_embed(patches)  # [B, n_patches, d_model]

        # Transformer
        z = self.encoder(z)  # [B, n_patches, d_model]

        # Flatten and forecast
        z = z.reshape(z.size(0), -1)  # [B, n_patches * d_model]
        return self.head(z)           # [B, horizon]


class PatchTSTTeacher(TeacherModel):
    """
    PatchTST teacher wrapper that follows the generic TeacherModel interface.
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        epochs: int,
        patch_len: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = PatchTST(
            lookback=lookback,
            horizon=horizon,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        ).to(self.device)

    def fit(self, series):
        """
        Train PatchTST on a single univariate series.
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