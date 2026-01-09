import pandas as pd
import numpy as np


def load_series(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df["price"].values.astype("float32")


def normalize(series):
    mean = series.mean()
    std = series.std() + 1e-6
    return (series - mean) / std, mean, std


def split_series(series, val_size=24, test_size=24):
    train = series[: -(val_size + test_size)]
    val = series[-(val_size + test_size) : -test_size]
    test = series[-test_size:]
    return train, val, test


def make_windows(series, lookback=24, horizon=6):
    X, Y = [], []
    for i in range(lookback, len(series) - horizon + 1):
        X.append(series[i - lookback : i])
        Y.append(series[i : i + horizon])
    return np.array(X), np.array(Y)


def compute_regime_scores(series, lookback, beta=0.3, tau=0.5):
    """
    Bounded regime-instability score using rolling volatility.
    Returned weights are in [1-beta, 1+beta].
    """
    raw = []
    for i in range(lookback, len(series)):
        window = series[i - lookback : i]
        raw.append(window.std())

    raw = np.array(raw, dtype="float32")
    raw = raw / (raw.mean() + 1e-6)  # normalize around 1

    # bounded, smooth transform
    scores = 1.0 + beta * np.tanh((raw - 1.0) / tau)
    return scores.astype("float32")


def augment_data(X, Y, noise_std=0.02, scale_range=(0.95, 1.05)):
    """
    Apply data augmentation to time series windows.

    Args:
        X: Input windows [N, lookback]
        Y: Target windows [N, horizon]
        noise_std: Standard deviation of Gaussian noise to add
        scale_range: Range for random scaling factor

    Returns:
        Augmented X and Y
    """
    X_aug = []
    Y_aug = []

    # Original data
    X_aug.append(X)
    Y_aug.append(Y)

    # Augmentation 1: Add small Gaussian noise
    noise_x = np.random.normal(0, noise_std, X.shape).astype("float32")
    noise_y = np.random.normal(0, noise_std, Y.shape).astype("float32")
    X_aug.append(X + noise_x)
    Y_aug.append(Y + noise_y)

    # Augmentation 2: Random scaling
    scale = np.random.uniform(scale_range[0], scale_range[1], (len(X), 1)).astype("float32")
    X_aug.append(X * scale)
    Y_aug.append(Y * scale)

    return np.vstack(X_aug), np.vstack(Y_aug)