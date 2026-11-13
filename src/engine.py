"""
src/engine.py
=============
Loop inti: training, validation, testing.
Dipakai oleh train_teacher.py, train_student.py, run_benchmark_models.py, evaluate.py.

Ada 2 mode training:
  1. train_standard(...)  -> untuk Teacher & model baseline (loss = MAE biasa)
  2. train_with_kd(...)   -> untuk Student (loss = DistillationLoss, teacher di-freeze)
"""

import numpy as np
import torch
import torch.nn as nn

import math
from utils import EarlyStopper, save_checkpoint, CSVLogger, Timer
from src.distillation import DistillationLoss


# --------------------------------------------------------------------------- #
# Helper: pilih loss function & LR scheduler sesuai cfg.train
# --------------------------------------------------------------------------- #
def build_hard_loss(cfg):
    """Loss untuk label asli (dipakai Teacher standar & komponen MAE di KD)."""
    if cfg.train.loss_type == "huber":
        return nn.SmoothL1Loss(beta=cfg.train.huber_delta)
    return nn.L1Loss()


def build_scheduler(optimizer, cfg, num_epochs: int):
    """
    Mengembalikan (scheduler, step_mode) dengan step_mode salah satu:
      "plateau" -> scheduler.step(val_loss) dipanggil tiap akhir epoch
      "epoch"   -> scheduler.step() dipanggil tiap akhir epoch (tanpa argumen)
    """
    if cfg.train.lr_scheduler_type == "cosine_warmup":
        warmup_epochs = max(cfg.train.warmup_epochs, 1)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, "epoch"

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.train.lr_scheduler_factor,
        patience=cfg.train.lr_scheduler_patience,
    )
    return scheduler, "plateau"


# --------------------------------------------------------------------------- #
# Metrik evaluasi
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    # akurasi "toleransi": persentase prediksi dgn error <= 2 bpm (umum dipakai di literatur RR)
    acc_2bpm = float(np.mean(np.abs(y_true - y_pred) <= 2.0) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "Acc@2bpm(%)": acc_2bpm}


# --------------------------------------------------------------------------- #
# STANDARD TRAINING (Teacher / model baseline)
# --------------------------------------------------------------------------- #
def run_epoch_standard(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_samples = 0.0, 0
    all_true, all_pred = [], []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(n_samples, 1)
    return metrics


def train_standard(model, train_loader, val_loader, cfg, device, run_name: str,
                    num_epochs: int, log_path: str, ckpt_path: str):
    """Dipakai untuk Teacher training maupun baseline model training."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate,
                                  weight_decay=cfg.train.weight_decay)
    scheduler, step_mode = build_scheduler(optimizer, cfg, num_epochs)
    criterion = build_hard_loss(cfg)
    early_stopper = EarlyStopper(patience=cfg.train.early_stopping_patience)

    logger = CSVLogger(log_path, fieldnames=[
        "epoch", "train_loss", "train_mae", "val_loss", "val_mae", "val_rmse", "lr",
    ])

    best_val_mae = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_m = run_epoch_standard(model, train_loader, device, criterion, optimizer)
        val_m = run_epoch_standard(model, val_loader, device, criterion, optimizer=None)

        current_lr = optimizer.param_groups[0]["lr"]
        if step_mode == "plateau":
            scheduler.step(val_m["loss"])
        else:
            scheduler.step()

        logger.log({
            "epoch": epoch,
            "train_loss": train_m["loss"], "train_mae": train_m["MAE"],
            "val_loss": val_m["loss"], "val_mae": val_m["MAE"], "val_rmse": val_m["RMSE"],
            "lr": current_lr,
        })
        print(f"[{run_name}] Epoch {epoch:03d} | train_MAE={train_m['MAE']:.3f} "
              f"| val_MAE={val_m['MAE']:.3f} | val_RMSE={val_m['RMSE']:.3f}")

        if val_m["MAE"] < best_val_mae:
            best_val_mae = val_m["MAE"]
            save_checkpoint(model, ckpt_path, extra={"epoch": epoch, "val_mae": best_val_mae})

        if early_stopper.step(val_m["loss"]):
            print(f"[{run_name}] Early stopping di epoch {epoch}")
            break

    logger.close()
    return best_val_mae


# --------------------------------------------------------------------------- #
# KNOWLEDGE DISTILLATION TRAINING (Student)
# --------------------------------------------------------------------------- #
def run_epoch_kd(student, teacher, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    student.train() if is_train else student.eval()
    teacher.eval()  # teacher SELALU frozen/eval

    total_loss, n_samples = 0.0, 0
    all_true, all_pred = [], []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                y_teacher = teacher(x)

            y_student = student(x)
            loss, loss_distill, loss_student = criterion(y_student, y_teacher, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(y_student.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(n_samples, 1)
    return metrics


def train_with_kd(student, teacher, train_loader, val_loader, cfg, device,
                   run_name: str, num_epochs: int, log_path: str, ckpt_path: str):
    student.to(device)
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.train.learning_rate,
                                  weight_decay=cfg.train.weight_decay)
    scheduler, step_mode = build_scheduler(optimizer, cfg, num_epochs)
    criterion = DistillationLoss(alpha=cfg.kd.alpha, hard_loss_type=cfg.train.loss_type,
                                  huber_delta=cfg.train.huber_delta)
    early_stopper = EarlyStopper(patience=cfg.train.early_stopping_patience)

    logger = CSVLogger(log_path, fieldnames=[
        "epoch", "train_loss", "train_mae", "val_loss", "val_mae", "val_rmse", "lr",
    ])

    best_val_mae = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_m = run_epoch_kd(student, teacher, train_loader, device, criterion, optimizer)
        val_m = run_epoch_kd(student, teacher, val_loader, device, criterion, optimizer=None)

        current_lr = optimizer.param_groups[0]["lr"]
        if step_mode == "plateau":
            scheduler.step(val_m["loss"])
        else:
            scheduler.step()

        logger.log({
            "epoch": epoch,
            "train_loss": train_m["loss"], "train_mae": train_m["MAE"],
            "val_loss": val_m["loss"], "val_mae": val_m["MAE"], "val_rmse": val_m["RMSE"],
            "lr": current_lr,
        })
        print(f"[{run_name}] Epoch {epoch:03d} | train_MAE={train_m['MAE']:.3f} "
              f"| val_MAE={val_m['MAE']:.3f} | val_RMSE={val_m['RMSE']:.3f}")

        if val_m["MAE"] < best_val_mae:
            best_val_mae = val_m["MAE"]
            save_checkpoint(student, ckpt_path, extra={"epoch": epoch, "val_mae": best_val_mae})

        if early_stopper.step(val_m["loss"]):
            print(f"[{run_name}] Early stopping di epoch {epoch}")
            break

    logger.close()
    return best_val_mae


# --------------------------------------------------------------------------- #
# TESTING (dipakai untuk semua jenis model: teacher, student, baseline)
# --------------------------------------------------------------------------- #
def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            all_true.append(y.numpy())
            all_pred.append(pred)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return compute_metrics(y_true, y_pred)
