import torch
import numpy as np

from core.dataset import *
from teachers.dlinear import DLinearTeacher
from teachers.patchtst import PatchTSTTeacher
from students.mlp import MLPStudent
from utils.uncertainty import compute_weights
from core.kd_trainer import train_student_kd
from core.metrics import mae, rmse, smape, horizon_metrics

DATA_PATH = "data/Wfp_rice.csv"
LOOKBACK = 24
HORIZON = 6
VAL_SIZE = 24
TEST_SIZE = 24

series = load_series(DATA_PATH)
series, mean, std = normalize(series)

train, val, test = split_series(series)

# windows for training (pure train only)
Xtr, Ytr = make_windows(train, lookback=LOOKBACK, horizon=HORIZON)

# windows for validation (train tail provides context)
val_context = np.concatenate([train[-LOOKBACK:], val])
Xva, Yva = make_windows(val_context, lookback=LOOKBACK, horizon=HORIZON)

# windows for test (train+val tail provides context)
test_context = np.concatenate([np.concatenate([train, val])[-LOOKBACK:], test])
Xte, Yte = make_windows(test_context, lookback=LOOKBACK, horizon=HORIZON)

teachers = [DLinearTeacher(), PatchTSTTeacher()]
tr_preds, va_preds = [], []

for t in teachers:
    t.fit(train)
    tr_preds.append(t.predict(Xtr))
    va_preds.append(t.predict(Xva))

weights = compute_weights(Yva, va_preds)
student = train_student_kd(Xtr, Ytr, tr_preds, weights, MLPStudent())

# Evaluate on test set
student.eval()
with torch.no_grad():
    student_preds = student(
        torch.tensor(Xte).unsqueeze(-1)
    ).cpu().numpy()

# de-normalize
student_preds = student_preds * std + mean
Yte = Yte * std + mean

print("Overall MAE:", mae(Yte, student_preds))
print("Overall RMSE:", rmse(Yte, student_preds))
print("Overall sMAPE:", smape(Yte, student_preds))

print("\nHorizon-wise metrics:")
for h, m in horizon_metrics(Yte, student_preds).items():
    print(h, m)

print("Training complete.")
