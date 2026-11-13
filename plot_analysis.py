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

def extract_rr_labels(data_dict):
    for key in ["labels", "label", "y", "targets", "target", "rr", "rrs", "y_true"]:
        if key in data_dict:
            val = data_dict[key]
            if isinstance(val, torch.Tensor):
                return val.cpu().numpy().flatten()
            return np.array(val).flatten()
    raise KeyError(f"Key label tidak ditemukan di: {list(data_dict.keys())}")

def plot_dataset_distributions(cfg, show_bidmc=True, show_capnobase=True, save_dir="results/plots"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2 if (show_bidmc and show_capnobase) else 1, figsize=(14 if (show_bidmc and show_capnobase) else 7, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    ax_idx = 0

    if show_bidmc and os.path.exists(cfg.BIDMC_PROCESSED_PATH):
        ax = axes[ax_idx]
        data_bidmc = torch.load(cfg.BIDMC_PROCESSED_PATH, weights_only=False)
        rr_bidmc = extract_rr_labels(data_bidmc)
        
        ax.hist(rr_bidmc, bins=30, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
        ax.axvline(np.mean(rr_bidmc), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(rr_bidmc):.2f} bpm")
        ax.axvline(np.median(rr_bidmc), color="green", linestyle=":", linewidth=2, label=f"Median: {np.median(rr_bidmc):.2f} bpm")
        ax.axvspan(0, 12, color="orange", alpha=0.15, label="Bradypnea (<12)")
        ax.axvspan(12, 20, color="blue", alpha=0.08, label="Normal (12-20)")
        ax.axvspan(20, 50, color="red", alpha=0.15, label="Tachypnea (>20)")
        
        ax.set_title(f"Distribusi RR BIDMC (N = {len(rr_bidmc)} Segmen)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Laju Napas / RR (bpm)", fontsize=11)
        ax.set_ylabel("Kerapatan Probabilitas (Density)", fontsize=11)
        ax.set_xlim(5, 45)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        ax_idx += 1

    if show_capnobase and os.path.exists(cfg.CAPNOBASE_PROCESSED_PATH):
        ax = axes[ax_idx]
        data_capno = torch.load(cfg.CAPNOBASE_PROCESSED_PATH, weights_only=False)
        rr_capno = extract_rr_labels(data_capno)
        
        ax.hist(rr_capno, bins=30, color="#2ca02c", edgecolor="black", alpha=0.7, density=True)
        ax.axvline(np.mean(rr_capno), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(rr_capno):.2f} bpm")
        ax.axvline(np.median(rr_capno), color="blue", linestyle=":", linewidth=2, label=f"Median: {np.median(rr_capno):.2f} bpm")
        ax.axvspan(0, 12, color="orange", alpha=0.15)
        ax.axvspan(12, 20, color="blue", alpha=0.08)
        ax.axvspan(20, 50, color="red", alpha=0.15)
        
        ax.set_title(f"Distribusi RR CapnoBase (N = {len(rr_capno)} Segmen)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Laju Napas / RR (bpm)", fontsize=11)
        ax.set_ylabel("Kerapatan Probabilitas (Density)", fontsize=11)
        ax.set_xlim(5, 45)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "dataset_rr_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Histogram tersimpan di: {save_path}")
    plt.show()

def plot_subject_time_series(model, test_loader, device, model_name="Student (KD)", is_transformer=True, num_samples=120, use_vmd=False, save_dir="results/plots"):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    y_trues, y_preds = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            if use_vmd:
                xb, _ = apply_vmd_batch(xb)
            out = model(xb)[0] if is_transformer else model(xb)
            y_preds.extend(out.cpu().numpy().flatten())
            y_trues.extend(yb.numpy().flatten())

    y_trues = np.array(y_trues)[:num_samples]
    y_preds = np.array(y_preds)[:num_samples]
    time_steps = np.arange(len(y_trues)) * 2.0  # Step 2 detik

    metrics = compute_metrics(y_trues, y_preds)

    plt.figure(figsize=(13, 5))
    plt.plot(time_steps, y_trues, label="Ground Truth", color="black", linewidth=2.0, marker="o", markersize=3)
    plt.plot(time_steps, y_preds, label=f"Prediksi {model_name}", color="#1f77b4", linewidth=1.8, linestyle="--", marker="x", markersize=4)
    plt.fill_between(time_steps, y_trues - 2, y_trues + 2, color="gray", alpha=0.2, label=r"Toleransi ($\pm 2$ bpm)")

    plt.title(f"Time-Series Tracking RR: {model_name}\nMAE: {metrics['MAE']:.3f} bpm | PCC: {metrics['PCC']:.3f} | Acc@2bpm: {metrics['Acc@2bpm(%)']:.1f}%", fontsize=12, fontweight="bold")
    plt.xlabel("Waktu Relatif (Detik)", fontsize=11, fontweight="bold")
    plt.ylabel("Laju Napas (bpm)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(save_dir, f"timeseries_tracking_{clean_name}.png")
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Grafik runut waktu tersimpan di: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_hist", action="store_true", default=False)
    parser.add_argument("--plot_tracking", action="store_true", default=False)
    parser.add_argument("--model", type=str, default="STUDENT_KD")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--use_vmd", action="store_true", default=False)
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device(cfg.train.device)

    if args.plot_hist:
        plot_dataset_distributions(cfg)

    if args.plot_tracking:
        _, _, test_loader = get_dataloaders(cfg, dataset_name=cfg.TRAIN_DATASET)
        target = args.model.upper()
        is_trans = True

        if target == "TEACHER":
            model = build_transformer(cfg.teacher_model, is_student=False)
            ckpt = os.path.join(cfg.CHECKPOINT_DIR, "teacher_best.pth")
            name = "Teacher (Transformer)"
        elif target == "STUDENT_KD":
            model = build_transformer(cfg.student_model, is_student=True)
            ckpt = os.path.join(cfg.CHECKPOINT_DIR, "student_kd_best.pth")
            name = "Student (KD)"
        elif target == "STUDENT_SCRATCH":
            model = build_transformer(cfg.student_model, is_student=True)
            ckpt = os.path.join(cfg.CHECKPOINT_DIR, "student_scratch_best.pth")
            name = "Student (Scratch)"
        elif target == "TRANSRR":
            model = TransRR()
            ckpt = os.path.join(cfg.CHECKPOINT_DIR, "transrr_best.pth")
            name = "TransRR Baseline"
            is_trans = False
        elif target == "HYBRID_CNN_LSTM":
            model = HybridCNNLSTM()
            ckpt = os.path.join(cfg.CHECKPOINT_DIR, "hybrid_cnn_lstm_best.pth")
            name = "Hybrid CNN-LSTM"
            is_trans = False
        elif target == "CONVMIXER":
            model = ConvMixer()
            ckpt = os.path.join(cfg.CHECKPOINT_DIR, "convmixer_best.pth")
            name = "ConvMixer"
            is_trans = False
        else:
            model = Resptrack1DCNN()
            ckpt = os.path.join(cfg.CHECKPOINT_DIR, "resptrack_1dcnn_best.pth")
            name = "Resptrack 1D-CNN"
            is_trans = False

        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
            plot_subject_time_series(model, test_loader, device, model_name=name, is_transformer=is_trans, num_samples=args.samples, use_vmd=args.use_vmd)
        else:
            print(f"[WARN] Checkpoint {ckpt} tidak ditemukan.")

if __name__ == "__main__":
    main()