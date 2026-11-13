"""
run_ablation_preprocessing.py
==============================
Skenario Pengujian #1: menguji pengaruh tahapan preprocessing (bandpass
filter, normalisasi, quality filtering) terhadap performa model.

Melatih model TEACHER (cukup teacher, karena tujuannya menguji efek data,
bukan efek kompresi) pada 4 kombinasi preprocessing:
    1. no_preprocessing     : tanpa filter, tanpa normalisasi, tanpa quality-filter
    2. filter_only          : + bandpass filter
    3. filter_norm          : + bandpass filter + normalisasi
    4. full_preprocessing   : + bandpass filter + normalisasi + quality filtering (full)

Hasil dirangkum di results/ablation_preprocessing_summary.csv

Cara menjalankan:
    python run_ablation_preprocessing.py
    python run_ablation_preprocessing.py --epochs 50    # mempercepat uji coba
"""

import argparse
import csv
import os
import torch

from config import get_config
from utils import set_seed, get_device
from src.dataset import get_dataloaders
from src.models.transformer import build_transformer
from src.engine import train_standard, test_model

PRESETS = ["no_preprocessing", "filter_only", "filter_norm", "full_preprocessing"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--presets", nargs="+", default=PRESETS)
    return p.parse_args()


def main():
    args = parse_args()
    results = []

    for preset in args.presets:
        print(f"\n{'='*70}\n[ABLATION] Menjalankan preset: {preset}\n{'='*70}")
        cfg = get_config(preset=preset)
        if args.epochs:
            cfg.train.num_epochs_teacher = args.epochs

        set_seed(cfg.split.random_seed)
        device = get_device(cfg.train.device)

        # cache preprocessing berbeda otomatis per preset (lihat src/dataset.py _cache_key)
        train_loader, val_loader, test_loader = get_dataloaders(cfg)

        model = build_transformer(cfg.teacher_model)
        log_path = os.path.join(cfg.paths.log_dir, f"ablation_{preset}_train_log.csv")
        ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"ablation_{preset}_best.pt")

        train_standard(model, train_loader, val_loader, cfg, device,
                        run_name=f"ABLATION-{preset}",
                        num_epochs=cfg.train.num_epochs_teacher,
                        log_path=log_path, ckpt_path=ckpt_path)

        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        metrics = test_model(model, test_loader, device)
        metrics["preset"] = preset
        results.append(metrics)
        print(f"[ABLATION] Hasil test {preset}: {metrics}")

    summary_path = "results/ablation_preprocessing_summary.csv"
    os.makedirs("results", exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["preset", "MAE", "RMSE", "MSE", "Acc@2bpm(%)"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n[DONE] Ringkasan ablation study disimpan di: {summary_path}")


if __name__ == "__main__":
    main()
