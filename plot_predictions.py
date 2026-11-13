import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import get_config
from src.dataset import get_dataloaders
from src.models.transformer import build_transformer
from src.models.baselines import TransRR, HybridCNNLSTM, ConvMixer, Resptrack1DCNN
from src.efficiency import compute_metrics
from src.vmd_utils import apply_vmd_batch

def main():
    parser = argparse.ArgumentParser(description="Visualisasi Prediction Plot (Scatter & Tracking)")
    parser.add_argument("--target", type=str, default="STUDENT_KD", 
                        choices=["TEACHER", "STUDENT_KD", "STUDENT_SCRATCH", "TRANSRR", "HYBRID_CNN_LSTM", "CONVMIXER", "RESPTRACK_1DCNN"])
    parser.add_argument("--use_vmd", action="store_true", default=False)
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device(cfg.train.device)
    target = args.target.upper()
    is_transformer = True

    if target == "TEACHER":
        model = build_transformer(cfg.teacher_model, is_student=False)
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "teacher_best.pth")
        model_title = "Teacher (Transformer)"
    elif target == "STUDENT_KD":
        model = build_transformer(cfg.student_model, is_student=True)
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "student_kd_best.pth")
        model_title = "Student (KD)"
    elif target == "STUDENT_SCRATCH":
        model = build_transformer(cfg.student_model, is_student=True)
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "student_scratch_best.pth")
        model_title = "Student (Scratch)"
    elif target == "TRANSRR":
        model = TransRR()
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "transrr_best.pth")
        model_title = "TransRR"
        is_transformer = False
    elif target == "HYBRID_CNN_LSTM":
        model = HybridCNNLSTM()
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "hybrid_cnn_lstm_best.pth")
        model_title = "Hybrid CNN-LSTM"
        is_transformer = False
    elif target == "CONVMIXER":
        model = ConvMixer()
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "convmixer_best.pth")
        model_title = "ConvMixer"
        is_transformer = False
    else:
        model = Resptrack1DCNN()
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "resptrack_1dcnn_best.pth")
        model_title = "Resptrack 1D-CNN"
        is_transformer = False

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint '{ckpt_path}' tidak ditemukan. Latih model terlebih dahulu.")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders(cfg, dataset_name=cfg.TRAIN_DATASET)

    y_trues, y_preds = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            if args.use_vmd or getattr(cfg, "USE_VMD", False):
                xb, _ = apply_vmd_batch(xb)
            out = model(xb)[0] if is_transformer else model(xb)
            y_preds.extend(out.cpu().numpy().flatten())
            y_trues.extend(yb.numpy().flatten())

    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    metrics = compute_metrics(y_trues, y_preds)

    os.makedirs(os.path.join(cfg.RESULTS_DIR, "plots"), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Scatter Plot
    ax1.scatter(y_trues, y_preds, color="#1f77b4", alpha=0.4, edgecolors="none", s=25)
    lims = [min(y_trues.min(), y_preds.min()) - 2, max(y_trues.max(), y_preds.max()) + 2]
    ax1.plot(lims, lims, color="red", linestyle="--", linewidth=1.5, label="Ideal (y = x)")
    ax1.fill_between(lims, np.array(lims) - 2, np.array(lims) + 2, color="gray", alpha=0.15, label=r"$\pm 2$ bpm")
    ax1.set_title(f"Scatter Plot: {model_title}\nMAE: {metrics['MAE']:.3f} | PCC: {metrics['PCC']:.3f}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Ground Truth RR (bpm)", fontsize=10)
    ax1.set_ylabel("Prediksi Model (bpm)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # 2. Tracking Plot (100 Sampel Pertama)
    n_show = min(100, len(y_trues))
    t_steps = np.arange(n_show) * 2.0
    ax2.plot(t_steps, y_trues[:n_show], label="Ground Truth", color="black", linewidth=1.8, marker="o", markersize=3)
    ax2.plot(t_steps, y_preds[:n_show], label="Prediksi", color="#1f77b4", linewidth=1.5, linestyle="--", marker="x", markersize=3)
    ax2.set_title(f"Tracking Plot: {model_title} (100 Sampel)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Waktu (Detik)", fontsize=10)
    ax2.set_ylabel("Laju Napas (bpm)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    save_file = os.path.join(cfg.RESULTS_DIR, "plots", f"prediction_plot_{target.lower()}.png")
    plt.savefig(save_file, dpi=300)
    print(f"[INFO] Plot hasil estimasi tersimpan di: {save_file}")
    plt.show()

if __name__ == "__main__":
    main()