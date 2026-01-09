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