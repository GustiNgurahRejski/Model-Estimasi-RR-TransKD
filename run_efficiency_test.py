"""
run_efficiency_test.py
========================
Skenario Pengujian #2 & #3:
  - Membandingkan performa (MAE dsb) Teacher vs Student secara terpisah.
  - Mengukur efisiensi komputasi (jumlah parameter, ukuran file model,
    inference latency) dalam simulasi kondisi wearable (batch_size=1,
    CPU-only, jumlah thread dibatasi).

Cara menjalankan:
    python run_efficiency_test.py \
        --teacher_ckpt checkpoints/teacher_best.pt \
        --student_ckpt checkpoints/student_best.pt \
        --preset full_preprocessing

    # tanpa simulasi wearable (kondisi normal, pakai GPU jika ada):
    python run_efficiency_test.py --teacher_ckpt ... --student_ckpt ... --no_wearable_sim

    # atur jumlah thread CPU yg disimulasikan (mis. mendekati SoC low-power):
    python run_efficiency_test.py --teacher_ckpt ... --student_ckpt ... --cpu_threads 1
"""

import argparse
import csv
import os
import torch

from config import get_config
from utils import set_seed, get_device
from src.dataset import get_dataloaders
from src.models.transformer import build_transformer
from src.engine import test_model
from src.efficiency import run_efficiency_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--no_wearable_sim", action="store_true",
                    help="matikan simulasi wearable, ukur di kondisi normal (mis. pakai GPU)")
    p.add_argument("--cpu_threads", type=int, default=None,
                    help="override jumlah thread CPU yang disimulasikan")
    p.add_argument("--wearable_batch_size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(preset=args.preset)
    cfg.efficiency.simulate_wearable = not args.no_wearable_sim
    if args.cpu_threads:
        cfg.efficiency.wearable_cpu_threads = args.cpu_threads
    if args.wearable_batch_size:
        cfg.efficiency.wearable_batch_size = args.wearable_batch_size

    set_seed(cfg.split.random_seed)
    device = get_device(cfg.train.device)

    _, _, test_loader = get_dataloaders(cfg)

    rows = []
    for name, ckpt, model_cfg in [
        ("teacher", args.teacher_ckpt, cfg.teacher_model),
        ("student", args.student_ckpt, cfg.student_model),
    ]:
        model = build_transformer(model_cfg)
        payload = torch.load(ckpt, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)

        perf = test_model(model, test_loader, device)
        eff = run_efficiency_report(model, cfg, model_name=name)

        row = {"model": name, **perf, **eff}
        rows.append(row)
        print(f"\n[{name.upper()}] Performa: {perf}")
        print(f"[{name.upper()}] Efisiensi: {eff}")

    # ringkasan perbandingan compression ratio
    if len(rows) == 2:
        t_params = rows[0]["num_parameters"]
        s_params = rows[1]["num_parameters"]
        t_lat = rows[0]["latency_mean_ms"]
        s_lat = rows[1]["latency_mean_ms"]
        t_size = rows[0]["model_size_kb"]
        s_size = rows[1]["model_size_kb"]
        print("\n" + "=" * 60)
        print("RINGKASAN PERBANDINGAN TEACHER vs STUDENT")
        print("=" * 60)
        print(f"Reduksi jumlah parameter : {t_params / s_params:.2f}x  "
              f"({t_params:,} -> {s_params:,})")
        print(f"Percepatan latency        : {t_lat / s_lat:.2f}x  "
              f"({t_lat:.3f} ms -> {s_lat:.3f} ms)")
        print(f"Reduksi ukuran model      : {t_size / s_size:.2f}x  "
              f"({t_size:.1f} KB -> {s_size:.1f} KB)")
        print(f"Selisih MAE               : {rows[1]['MAE'] - rows[0]['MAE']:+.4f} bpm "
              f"(student - teacher)")

    os.makedirs("results", exist_ok=True)
    summary_path = "results/efficiency_teacher_vs_student.csv"
    fieldnames = sorted(set().union(*[r.keys() for r in rows]))
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[DONE] Ringkasan disimpan di: {summary_path}")


if __name__ == "__main__":
    main()
