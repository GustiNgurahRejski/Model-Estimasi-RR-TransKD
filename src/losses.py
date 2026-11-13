import torch
import torch.nn as nn
import torch.nn.functional as F

class PearsonCorrelationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        pred_flat = y_pred.view(-1)
        true_flat = y_true.view(-1)

        vx = pred_flat - torch.mean(pred_flat)
        vy = true_flat - torch.mean(true_flat)

        cov = torch.sum(vx * vy)
        std_x = torch.sqrt(torch.sum(vx ** 2) + self.eps)
        std_y = torch.sqrt(torch.sum(vy ** 2) + self.eps)

        pcc = cov / (std_x * std_y)
        return 1.0 - pcc

class CorrelationAwareLoss(nn.Module):
    def __init__(self, alpha_pcc=2.0, hard_loss_type="mae", huber_delta=1.0):
        super().__init__()
        self.alpha_pcc = alpha_pcc
        self.pcc_loss = PearsonCorrelationLoss()
        self.hard_loss_type = hard_loss_type.lower()
        self.huber_delta = huber_delta

    def forward(self, y_pred, y_true):
        if self.hard_loss_type == "huber":
            hard_loss = F.huber_loss(y_pred, y_true, delta=self.huber_delta)
        elif self.hard_loss_type == "mse":
            hard_loss = F.mse_loss(y_pred, y_true)
        else:
            hard_loss = F.l1_loss(y_pred, y_true)

        pcc_penalty = self.pcc_loss(y_pred, y_true)
        return hard_loss + (self.alpha_pcc * pcc_penalty)

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha_kd=0.3, alpha_pcc=2.0, alpha_feat=0.1):
        super().__init__()
        self.alpha_kd = alpha_kd
        self.alpha_feat = alpha_feat
        self.task_loss = CorrelationAwareLoss(alpha_pcc=alpha_pcc)
        self.mse = nn.MSELoss()

    def forward(self, student_pred, teacher_pred, y_true, student_feat=None, teacher_feat=None):
        l_task = self.task_loss(student_pred, y_true)
        l_soft = self.mse(student_pred, teacher_pred.detach())
        loss = (1.0 - self.alpha_kd) * l_task + self.alpha_kd * l_soft

        if student_feat is not None and teacher_feat is not None:
            if student_feat.shape[1] != teacher_feat.shape[1]:
                t_pooled = F.adaptive_avg_pool1d(teacher_feat, student_feat.shape[2])
                s_pooled = F.adaptive_avg_pool1d(student_feat, student_feat.shape[2])
                l_feat = self.mse(s_pooled.mean(dim=1), t_pooled.mean(dim=1).detach())
            else:
                l_feat = self.mse(student_feat, teacher_feat.detach())
            loss += self.alpha_feat * l_feat

        return loss