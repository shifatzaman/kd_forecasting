
import torch
import torch.nn as nn
import numpy as np

def train_student_kd(X, Y, teacher_preds, weights, student, epochs=60, alpha0=0.7):
    X = torch.tensor(X).unsqueeze(-1)
    Y = torch.tensor(Y)
    T = torch.tensor(np.stack(teacher_preds,1))
    W = torch.tensor(weights).view(1,-1,1)

    opt = torch.optim.AdamW(student.parameters(), 1e-3)
    mae, mse = nn.L1Loss(), nn.MSELoss()

    for e in range(epochs):
        a = alpha0*(1-e/(epochs-1))
        p = student(X)
        t = (T*W).sum(1)
        loss = (1-a)*mae(p,Y)+a*mse(p,t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return student
