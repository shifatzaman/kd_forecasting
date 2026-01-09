import torch
import torch.nn as nn
import numpy as np


def train_student_kd(
    X,
    Y,
    teacher_preds,
    weights,
    student,
    epochs,
    regime_scores=None,
    alpha0=0.7,
):
    """
    Regime-aware knowledge distillation training.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    T = torch.tensor(np.stack(teacher_preds, axis=1), dtype=torch.float32).to(device)
    W = torch.tensor(weights, dtype=torch.float32).view(1, -1, 1).to(device)

    if regime_scores is not None:
        R = torch.tensor(regime_scores, dtype=torch.float32).view(-1, 1).to(device)
    else:
        R = torch.ones(len(X), 1, device=device)

    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    H = Y.shape[1]
    horizon_weights = torch.linspace(1.0, 0.5, H, device=device).view(1, -1)

    student.train()
    for e in range(epochs):
        alpha = alpha0 * (1 - e / (epochs - 1))

        preds = student(X)
        teacher_target = (T * W).sum(dim=1)

        # regime + horizon weighted losses
        loss_sup = (R * horizon_weights * torch.abs(preds - Y)).mean()
        loss_kd = (R * horizon_weights * (preds - teacher_target) ** 2).mean()

        loss = (1 - alpha) * loss_sup + alpha * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return student