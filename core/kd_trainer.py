import torch
import torch.nn.functional as F
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
    lr=1e-3,
    patience=30,
    val_data=None,
):
    """
    Knowledge distillation with bounded, KD-only regime weighting.
    Includes learning rate scheduling, gradient clipping, and early stopping.
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
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    H = Y.shape[1]
    horizon_weights = torch.linspace(1.0, 0.5, H, device=device).view(1, -1)

    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    student.train()
    for e in range(epochs):
        # Dynamic alpha: starts high, decreases gradually
        alpha = alpha0 * (1 - e / (epochs - 1))

        preds = student(X)
        teacher_target = (T * W).sum(dim=1)

        # Combined loss: MAE + Huber for robustness
        loss_mae = (horizon_weights * torch.abs(preds - Y)).mean()
        loss_huber = (horizon_weights * F.huber_loss(preds, Y, reduction='none', delta=1.0)).mean()
        loss_sup = 0.7 * loss_mae + 0.3 * loss_huber

        # KD loss (bounded regime weighting) - using MSE for smooth gradients
        loss_kd = (R * horizon_weights * (preds - teacher_target) ** 2).mean()

        loss = (1 - alpha) * loss_sup + alpha * loss_kd

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Restore best weights
                if best_state is not None:
                    student.load_state_dict(best_state)
                break

    # Load best model if we have it
    if best_state is not None:
        student.load_state_dict(best_state)

    return student