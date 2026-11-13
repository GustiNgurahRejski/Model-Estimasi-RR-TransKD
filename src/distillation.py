"""
src/distillation.py
===================
Loss function untuk tahap distilasi (Persamaan 3.1 pada tesis):

    L_total = alpha * MSE(y_student, y_teacher) + (1 - alpha) * MAE(y_student, y_true)

- MSE terhadap soft-target Teacher -> transfer pengetahuan terstruktur.
- MAE terhadap ground-truth        -> menjaga Student tetap berpegang pada label asli.
"""

import torch
import torch.nn as nn


class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, hard_loss_type: str = "mae",
                 huber_delta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        if hard_loss_type == "huber":
            self.hard_loss = nn.SmoothL1Loss(beta=huber_delta)
        else:
            self.hard_loss = nn.L1Loss()

    def forward(self, y_student: torch.Tensor, y_teacher: torch.Tensor,
                y_true: torch.Tensor):
        loss_distill = self.mse(y_student, y_teacher)
        loss_student = self.hard_loss(y_student, y_true)
        total = self.alpha * loss_distill + (1 - self.alpha) * loss_student
        return total, loss_distill.item(), loss_student.item()

class PearsonCorrelationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        
        cov = torch.sum(vx * vy)
        std_x = torch.sqrt(torch.sum(vx ** 2) + self.eps)
        std_y = torch.sqrt(torch.sum(vy ** 2) + self.eps)
        
        pcc = cov / (std_x * std_y)
        return 1.0 - pcc  # Bernilai 0 saat korelasi sempurna (+1.0)

class CorrelationAwareLoss(nn.Module):
    def __init__(self, alpha_pcc=2.0):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.pcc_loss = PearsonCorrelationLoss()
        self.alpha_pcc = alpha_pcc

    def forward(self, y_pred, y_true):
        mae = self.l1_loss(y_pred, y_true)
        pcc_penalty = self.pcc_loss(y_pred, y_true)
        return mae + self.alpha_pcc * pcc_penalty