import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1e-6):
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def smape(y_true, y_pred, eps=1e-6):
    return np.mean(
        2 * np.abs(y_true - y_pred) /
        (np.abs(y_true) + np.abs(y_pred) + eps)
    ) * 100

def horizon_metrics(y_true, y_pred):
    """
    y_true, y_pred: [N, H]
    """
    H = y_true.shape[1]
    out = {}

    for h in range(H):
        out[f"t+{h+1}"] = {
            "MAE": mae(y_true[:, h], y_pred[:, h]),
            "RMSE": rmse(y_true[:, h], y_pred[:, h]),
            "sMAPE": smape(y_true[:, h], y_pred[:, h])
        }
    return out