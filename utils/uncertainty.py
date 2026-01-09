
import numpy as np

def uncertainty(preds):
    return np.mean(np.var(preds, axis=0))

def compute_weights(y_val, teacher_preds):
    maes, uncs = [], []
    for p in teacher_preds:
        maes.append(np.mean(np.abs(y_val - p)))
        uncs.append(uncertainty(p))
    raw = 1 / ((np.array(maes)+1e-6)*(np.array(uncs)+1e-6))
    return raw / raw.sum()
