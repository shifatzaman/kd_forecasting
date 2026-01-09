
from core.dataset import *
from teachers.dlinear import DLinearTeacher
from teachers.patchtst import PatchTSTTeacher
from students.mlp import MLPStudent
from utils.uncertainty import compute_weights
from core.kd_trainer import train_student_kd

DATA_PATH = "data/sample.csv"

series = load_series(DATA_PATH)
train, val, test = split_series(series)

Xtr, Ytr = make_windows(train)
Xva, Yva = make_windows(val)

teachers = [DLinearTeacher(), PatchTSTTeacher()]
tr_preds, va_preds = [], []

for t in teachers:
    t.fit(train)
    tr_preds.append(t.predict(Xtr))
    va_preds.append(t.predict(Xva))

weights = compute_weights(Yva, va_preds)
student = train_student_kd(Xtr, Ytr, tr_preds, weights, MLPStudent())

print("Training complete.")
