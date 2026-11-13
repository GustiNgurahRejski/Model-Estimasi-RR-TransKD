import os
import argparse
import torch
import numpy as np
from config import get_config
from src.dataset import get_dataloaders, get_rr_stratified_subsets, PPGDataset
from src.models.transformer import build_transformer
from src.models.baselines import TransRR, HybridCNNLSTM, ConvMixer, Resptrack1DCNN
from src.efficiency import compute_metrics, measure_inference_latency, count_parameters, compute_model_size_mb
from src.vmd_utils import apply_vmd_batch

def eval_model(model, loader, device, is_transformer=False, use_vmd=False):
    model.to(device)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if use_vmd:
                xb, _ = apply_vmd_batch(xb)
            out = model(xb)[0] if is_transformer else model(xb)
            preds.extend(out.cpu().numpy().flatten())
            trues.extend(yb.numpy().flatten())
    return compute_metrics(trues, preds)

def main():
    parser = argparse.ArgumentParser(description="Evaluasi Komprehensif Hipotesis 1 & 2")
    parser.add_argument("--use_vmd", action="store_true", default=False, help="Aktifkan VMD saat inferensi")
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device(cfg.train.device)
    use_vmd_active = args.use_vmd or getattr(cfg, "USE_VMD", False)

    print("==================================================================")
    print(f" PENGUJIAN KOMPREHENSIF MODEL RR (Simulasi Wearable: {cfg.SIMULATE_WEARABLE})")
    print(f" Dataset Aktif: {cfg.ACTIVE_DATASET} | Mode VMD: {use_vmd_active}")
    print("==================================================================")

    models = {
        "Teacher (Transformer)": (build_transformer(cfg.teacher_model, is_student=False), True, "teacher_best.pth"),
        "Student (KD)": (build_transformer(cfg.student_model, is_student=True), True, "student_kd_best.pth"),
        "Student (Scratch)": (build_transformer(cfg.student_model, is_student=True), True, "student_scratch_best.pth"),
        "TransRR": (TransRR(), False, "transrr_best.pth"),
        "Hybrid CNN-LSTM": (HybridCNNLSTM(), False, "hybrid_cnn_lstm_best.pth"),
        "ConvMixer": (ConvMixer(), False, "convmixer_best.pth"),
        "Resptrack 1D-CNN": (Resptrack1DCNN(), False, "resptrack_1dcnn_best.pth")
    }

    for name, (m, _, ckpt) in models.items():
        if ckpt is not None:
            p = os.path.join(cfg.CHECKPOINT_DIR, ckpt)
            if os.path.exists(p):
                m.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            else:
                print(f"[INFO] Checkpoint '{ckpt}' untuk {name} belum ditemukan.")

    _, _, test_loader = get_dataloaders(cfg, dataset_name=cfg.ACTIVE_DATASET)

    # -------------------------------------------------------------------------
    # 1. Ringkasan Performa & Efisiensi Hardware (Hipotesis 1)
    # -------------------------------------------------------------------------
    print(f"\n>>> 1. Ringkasan Performa & Efisiensi Model ({cfg.ACTIVE_DATASET} Test Set):")
    print(f"{'Model Name':<22} | {'MAE':<7} | {'RMSE':<7} | {'PCC':<7} | {'Acc@2bpm':<9} | {'Latency':<10} | {'Params':<9} | {'Size':<9}")
    print("-" * 100)
    for name, (m, is_t, _) in models.items():
        res = eval_model(m, test_loader, device, is_t, use_vmd=use_vmd_active)
        lat = measure_inference_latency(m, simulate_wearable=cfg.SIMULATE_WEARABLE)
        
        if use_vmd_active:
            dummy = torch.randn(1, 1, cfg.data.samples_per_window).to(device)
            _, vmd_lat = apply_vmd_batch(dummy)
            lat += vmd_lat

        m.to(device)
        params = count_parameters(m)
        sz = compute_model_size_mb(m)["size_kb"]
        print(f"{name:<22} | {res['MAE']:<7.3f} | {res['RMSE']:<7.3f} | {res['PCC']:<7.3f} | {res['Acc@2bpm(%)']:<8.1f}% | {lat:<7.2f} ms | {params:<9,} | {sz:<7.1f} KB")

    # -------------------------------------------------------------------------
    # 2. Uji Robust Stratifikasi Nilai RR pada Dataset CapnoBase (Hipotesis 2)
    # -------------------------------------------------------------------------
    capno_path = cfg.CAPNOBASE_PROCESSED_PATH
    if getattr(cfg, "TEST_RR_STRATIFICATION", True) and os.path.exists(capno_path):
        print("\n==================================================================")
        print(" >>> 2. Uji Robust Stratifikasi Nilai RR pada Dataset CapnoBase <<<")
        print("==================================================================")
        
        capno_data = torch.load(capno_path, weights_only=False)
        subsets = get_rr_stratified_subsets(capno_data["x"], capno_data["y"])
        
        table_labels = {"Bradypnea (<12 bpm)": "2A", "Normal (12-20 bpm)": "2B", "Tachypnea (>20 bpm)": "2C"}
        
        for group_name, (x_sub, y_sub) in subsets.items():
            t_id = table_labels.get(group_name, "2")
            print(f"\n--- TABEL {t_id}: Kelompok {group_name} ({len(y_sub)} sampel) ---")
            if len(y_sub) == 0:
                print("Tidak ada sampel pada rentang ini.")
                continue
            
            sub_loader = torch.utils.data.DataLoader(PPGDataset(x_sub, y_sub), batch_size=64, shuffle=False)
            print(f"{'Model Name':<22} | {'MAE (bpm)':<10} | {'RMSE (bpm)':<10} | {'PCC':<8} | {'Acc@2bpm':<10} | Status (<2 bpm)")
            print("-" * 90)
            for name, (m, is_t, _) in models.items():
                res = eval_model(m, sub_loader, device, is_t, use_vmd=use_vmd_active)
                status = "[Lolos Klinis]" if res['MAE'] <= 2.0 else "[Di Luar Batas]"
                print(f"{name:<22} | {res['MAE']:<10.3f} | {res['RMSE']:<10.3f} | {res['PCC']:<8.3f} | {res['Acc@2bpm(%)']:<9.1f}% | {status}")

if __name__ == "__main__":
    main()