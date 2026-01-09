# sweep.py
import itertools
import numpy as np
import pandas as pd
import torch

from core.dataset import load_series, split_series, make_windows, normalize
from core.metrics import mae, rmse, smape, horizon_metrics
from core.kd_trainer import train_student_kd
from utils.uncertainty import compute_weights

from teachers.dlinear import DLinearTeacher
from teachers.patchtst import PatchTSTTeacher
from students.mlp import MLPStudent

# -------------------
# EDIT THESE
# -------------------
DATA_PATH = "data/Wfp_rice.csv"
HORIZON = 6
VAL_SIZE = 24
TEST_SIZE = 24
SEED = 42

LOOKBACKS = [36, 48, 60]
PATCH_LENS = [4]

TEACHER_SETS = [
    ("DLinear",),
    ("DLinear", "PatchTST"),
]

ALPHA0_LIST = [0.5, 0.7]

TEACHER_EPOCHS_LIST = [200]
STUDENT_EPOCHS_LIST = [200]

# -------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_teachers(names, lookback, horizon, teacher_epochs, patch_len):
    teachers = []
    for n in names:
        if n == "DLinear":
            teachers.append(DLinearTeacher(lookback=lookback, horizon=horizon, epochs=teacher_epochs))
        elif n == "PatchTST":
            teachers.append(PatchTSTTeacher(
                lookback=lookback, horizon=horizon, epochs=teacher_epochs, patch_len=patch_len
            ))
        else:
            raise ValueError(f"Unknown teacher: {n}")
    return teachers

def run_one(lookback, patch_len, teacher_epochs, student_epochs, alpha0, teacher_set):
    if "PatchTST" in teacher_set and lookback % patch_len != 0:
        return None  # invalid config

    # load + normalize
    series = load_series(DATA_PATH)
    series, mean, std = normalize(series)

    # split
    train, val, test = split_series(series, val_size=VAL_SIZE, test_size=TEST_SIZE)

    # context-aware windows
    val_ctx = np.concatenate([train[-lookback:], val])
    test_ctx = np.concatenate([np.concatenate([train, val])[-lookback:], test])

    Xtr, Ytr = make_windows(train, lookback=lookback, horizon=HORIZON)
    Xva, Yva = make_windows(val_ctx, lookback=lookback, horizon=HORIZON)
    Xte, Yte = make_windows(test_ctx, lookback=lookback, horizon=HORIZON)

    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        return None

    # teachers
    teachers = build_teachers(teacher_set, lookback, HORIZON, teacher_epochs, patch_len)

    tr_preds, va_preds = [], []
    for t in teachers:
        t.fit(train)
        tr_preds.append(t.predict(Xtr))
        va_preds.append(t.predict(Xva))

    weights = compute_weights(Yva, va_preds)

    # student
    student = MLPStudent(lookback=lookback, horizon=HORIZON)
    student = train_student_kd(
        X=Xtr, Y=Ytr,
        teacher_preds=tr_preds,
        weights=weights,
        student=student,
        epochs=student_epochs,
        alpha0=alpha0,
    )

    # evaluate
    student.eval()
    with torch.no_grad():
        preds_norm = student(torch.tensor(Xte, dtype=torch.float32).unsqueeze(-1)).cpu().numpy()

    # de-normalize to report in original units
    preds = preds_norm * std + mean
    Yte_denorm = Yte * std + mean

    overall = {
        "MAE": float(mae(Yte_denorm, preds)),
        "RMSE": float(rmse(Yte_denorm, preds)),
        "sMAPE": float(smape(Yte_denorm, preds)),
    }

    hm = horizon_metrics(Yte_denorm, preds)
    # record horizon MAEs
    for k, v in hm.items():
        overall[f"{k}_MAE"] = float(v["MAE"])
        overall[f"{k}_sMAPE"] = float(v["sMAPE"])

    overall.update({
        "lookback": lookback,
        "horizon": HORIZON,
        "patch_len": patch_len,
        "teacher_epochs": teacher_epochs,
        "student_epochs": student_epochs,
        "alpha0": alpha0,
        "teachers": "+".join(teacher_set),
        "teacher_weights": ",".join([f"{w:.4f}" for w in weights]),
    })
    return overall

def main():
    set_seed(SEED)
    rows = []
    grid = itertools.product(
        LOOKBACKS, PATCH_LENS, TEACHER_EPOCHS_LIST, STUDENT_EPOCHS_LIST, ALPHA0_LIST, TEACHER_SETS
    )

    for lookback, patch_len, te, se, a0, tset in grid:
        res = run_one(lookback, patch_len, te, se, a0, tset)
        if res is not None:
            rows.append(res)
            print("OK:", res["teachers"], "L", lookback, "P", patch_len, "MAE", res["MAE"], "sMAPE", res["sMAPE"])

    df = pd.DataFrame(rows).sort_values(["MAE", "sMAPE"]).reset_index(drop=True)
    df.to_csv("sweep_results.csv", index=False)
    print("\nSaved: sweep_results.csv")
    print(df.head(10)[["teachers","lookback","patch_len","teacher_epochs","student_epochs","alpha0","MAE","sMAPE"]])

if __name__ == "__main__":
    main()