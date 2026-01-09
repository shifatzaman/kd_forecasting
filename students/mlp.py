import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


class MLPStudent(nn.Module):
    def __init__(self, lookback=24, horizon=6, hidden_dim=256, n_blocks=3, dropout=0.1):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(lookback, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks for deep feature learning
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])

        # Output head with multiple stages
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, horizon)
        )

    def forward(self, x):
        x = x.squeeze(-1)

        # Input projection
        x = self.input_proj(x)

        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)

        # Generate forecast
        return self.output_head(x)
