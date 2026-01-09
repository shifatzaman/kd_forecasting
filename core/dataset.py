
import pandas as pd
import numpy as np

def load_series(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df["price"].values.astype("float32")

def split_series(series, val_size=24, test_size=24):
    train = series[:-(val_size + test_size)]
    val   = series[-(val_size + test_size):-test_size]
    test  = series[-test_size:]
    return train, val, test

def make_windows(series, lookback=24, horizon=6):
    X, Y = [], []
    for i in range(lookback, len(series)-horizon+1):
        X.append(series[i-lookback:i])
        Y.append(series[i:i+horizon])
    return np.array(X), np.array(Y)

def normalize(series):
    mean = series.mean()
    std = series.std() + 1e-6
    return (series - mean) / std, mean, std

def compute_regime_scores(series, lookback):
    """
    Computes a regime-instability score per window using rolling volatility.
    Returns one score per (X, Y) window.
    """
    scores = []
    for i in range(lookback, len(series)):
        window = series[i - lookback : i]
        score = window.std()
        scores.append(score)

    scores = np.array(scores)

    # normalize to mean 1 for stability
    scores = scores / (scores.mean() + 1e-6)
    return scores.astype("float32")
