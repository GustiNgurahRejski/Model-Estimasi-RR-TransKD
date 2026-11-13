"""
src/preprocessing.py
=====================
Pipeline signal-conditioning terintegrasi untuk dataset BIDMC dan CapnoBase.
Mendukung format mentah CSV dan MATLAB (.mat).

Format data mentah:
    bidmc_XX_Signals.csv -> Kolom: 'Time [s]', 'PLETH', 'RESP', dst (fs=125Hz)
    bidmc_XX_Breaths.csv -> Kolom: 'ann1' / 'ann2' (indeks sample nafas 2 annotator, fs=125Hz)
"""

"""
src/preprocessing.py
=====================
Ekstraktor langsung file .mat CapnoBase menggunakan h5py murni.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import torch
import h5py
from scipy import signal as sps
from scipy.interpolate import interp1d

try:
    from config import get_config, Config
except ImportError:
    get_config = None
    Config = None


def interpolate_nan(ppg: np.ndarray) -> np.ndarray:
    if not np.any(np.isnan(ppg)):
        return ppg
    idx = np.arange(len(ppg))
    valid = ~np.isnan(ppg)
    if valid.sum() < 2:
        return np.nan_to_num(ppg)
    f = interp1d(idx[valid], ppg[valid], bounds_error=False, fill_value=(ppg[valid][0], ppg[valid][-1]))
    return f(idx)

def downsample(ppg: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return ppg
    return sps.decimate(ppg, factor, ftype="fir", zero_phase=True)

def bandpass_filter(ppg: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low = max(0.001, low_hz / nyq)
    high = min(0.999, high_hz / nyq)
    b, a = sps.butter(order, [low, high], btype="band")
    return sps.filtfilt(b, a, ppg)

def zscore_normalize(x: np.ndarray) -> np.ndarray:
    mu, sigma = np.mean(x), np.std(x)
    if sigma < 1e-12:
        return x - mu
    return (x - mu) / sigma


# --------------------------------------------------------------------------- #
# 1. MODUL BIDMC (CSV)
# --------------------------------------------------------------------------- #
def process_bidmc_dataset(cfg):
    raw_dir = getattr(cfg, "RAW_BIDMC_DIR", os.path.join("data", "raw", "bidmc_csv"))
    save_path = getattr(cfg, "BIDMC_PROCESSED_PATH", os.path.join("data", "processed", "bidmc_processed.pt"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*_Signals.csv")))
    if not files:
        return

    print("==================================================================")
    print(f" MEMPROSES DATASET BIDMC ({len(files)} Subjek)")
    print("==================================================================")

    segments = []
    fs_orig = 125.0
    fs_target = float(getattr(cfg.data, "fs_target", 25.0))
    ds_factor = max(1, int(round(fs_orig / fs_target)))
    win_sec = getattr(cfg.data, "window_seconds", 30.0)
    step_sec = getattr(cfg.data, "window_step_seconds", 2.0)
    samples_per_win = int(win_sec * fs_target)
    step_samples = int(step_sec * fs_target)
    win_orig = int(win_sec * fs_orig)
    step_orig = int(step_sec * fs_orig)

    for f_sig in files:
        try:
            base = os.path.basename(f_sig)
            sid = base.split("_")[1] if len(base.split("_")) >= 3 else base.split("_")[0]
            f_breath = f_sig.replace("_Signals.csv", "_Breaths.csv")
            if not os.path.exists(f_breath): continue

            sig_df = pd.read_csv(f_sig)
            breath_df = pd.read_csv(f_breath)
            
            pleth_col = next((c for c in sig_df.columns if "pleth" in c.lower()), sig_df.columns[1])
            ppg_raw = sig_df[pleth_col].values.astype(np.float64)
            
            ann_cols = [c for c in breath_df.columns if "ann" in c.lower()]
            if len(ann_cols) < 2: continue
            ann1 = breath_df[ann_cols[0]].dropna().values.astype(int)
            ann2 = breath_df[ann_cols[1]].dropna().values.astype(int)

            ppg = interpolate_nan(ppg_raw)
            ppg_ds = downsample(ppg, ds_factor)
            ppg_ds = bandpass_filter(ppg_ds, fs_target, cfg.data.bandpass_low_hz, cfg.data.bandpass_high_hz, cfg.data.bandpass_order)
            ppg_ds = zscore_normalize(ppg_ds)

            n_win = (len(ppg_ds) - samples_per_win) // step_samples + 1
            for w in range(max(0, n_win)):
                st, et = w * step_samples, w * step_samples + samples_per_win
                if et > len(ppg_ds): break
                x = ppg_ds[st:et]

                so, eo = w * step_orig, w * step_orig + win_orig
                n1 = np.sum((ann1 >= so) & (ann1 < eo))
                n2 = np.sum((ann2 >= so) & (ann2 < eo))
                rr1, rr2 = (n1 / win_sec) * 60.0, (n2 / win_sec) * 60.0

                if abs(rr1 - rr2) > 3.0: continue
                rr_label = (rr1 + rr2) / 2.0

                if cfg.data.rr_min_bpm <= rr_label <= cfg.data.rr_max_bpm and np.std(x) > 1e-4:
                    segments.append({"x": x.astype(np.float32), "y": np.float32(rr_label), "subject_id": sid})
        except Exception:
            pass

    if segments:
        x_t = torch.tensor(np.array([s["x"] for s in segments]), dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(np.array([s["y"] for s in segments]), dtype=torch.float32).view(-1, 1)
        subjs = [s["subject_id"] for s in segments]
        torch.save({"x": x_t, "y": y_t, "labels": y_t, "subject_ids": subjs}, save_path)
        print(f"[SUKSES] BIDMC selesai: {len(x_t):,} segmen\n")


# --------------------------------------------------------------------------- #
# 2. MODUL CAPNOBASE HDF5 v7.3 (H5PY DIRECT DATA EXTRACTION)
# --------------------------------------------------------------------------- #
def load_capnobase_h5py_pure(mat_path: str):
    """
    Membaca file .mat HDF5 v7.3 CapnoBase.

    CATATAN PENTING (hasil inspeksi langsung struktur file):
    Struktur asli CapnoBase TIDAK menggunakan HDF5 reference/cell-array
    untuk field-field ini -- semuanya dataset numerik langsung:

        signal/pleth/y                -> (1, N) float64, sinyal PPG mentah
        param/samplingrate/pleth      -> (1, 1) float64, fs asli (Hz), umumnya 300.0
        labels/co2/startinsp/x        -> (n_breaths, 1) float64, indeks SAMPEL
                                          (bukan detik) awal tiap inspirasi -> dipakai
                                          sebagai penanda 1 napas / breath marker.

    Kode versi sebelumnya mengasumsikan field-field ini berupa HDF5 reference
    yang perlu di-dereference (f[ref]), sehingga saat dataset (1,1) di-.item()
    hasilnya adalah tuple/scalar Python biasa (bukan h5py.Reference), dan
    f[tuple] gagal dengan:
        "Accessing a group is done with bytes or str, not <class 'tuple'>"
    Perbaikan: baca langsung tanpa proses dereference sama sekali.
    """
    try:
        with h5py.File(mat_path, 'r') as f:
            # 1. Sinyal PPG mentah
            ppg_raw = np.array(f['signal']['pleth']['y'][()]).flatten().astype(np.float64)

            # 2. Sampling rate asli sinyal pleth
            fs_orig = float(np.array(f['param']['samplingrate']['pleth'][()]).flatten()[0])

            # 3. Anotasi breath: indeks sampel awal inspirasi (startinsp)
            breath_idx = np.array([])
            if 'labels' in f and 'co2' in f['labels'] and 'startinsp' in f['labels']['co2']:
                breath_idx = np.array(f['labels']['co2']['startinsp']['x'][()]).flatten().astype(np.float64)

            if len(ppg_raw) > 1000 and len(breath_idx) > 0:
                return ppg_raw, fs_orig, breath_idx
    except Exception as e:
        print(f"  [WARN] Gagal h5py pada {os.path.basename(mat_path)}: {e}")

    return None, 300.0, np.array([])


def process_capnobase_dataset(cfg):
    raw_dir = getattr(cfg, "RAW_CAPNOBASE_DIR", os.path.join("data", "raw", "capnobase"))
    save_path = getattr(cfg, "CAPNOBASE_PROCESSED_PATH", os.path.join("data", "processed", "capnobase_processed.pt"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.mat"), recursive=True))
    if not mat_files:
        mat_files = sorted(glob.glob(os.path.join(raw_dir, "*.mat")))

    print("==================================================================")
    print(f" MEMPROSES DATASET CAPNOBASE BENCHMARK ({len(mat_files)} File .MAT)")
    print("==================================================================")

    if not mat_files:
        print(f"[PERINGATAN] Tidak ada file .mat di {raw_dir}.")
        return

    segments = []
    win_sec = getattr(cfg.data, "window_seconds", 30.0)
    fs_target = getattr(cfg.data, "fs_target", 25.0)
    samples_per_win = int(win_sec * fs_target)
    step_sec = getattr(cfg.data, "window_step_seconds", 2.0)
    step_samples = int(step_sec * fs_target)

    valid_count = 0
    for mat_path in mat_files:
        try:
            base_name = os.path.basename(mat_path)
            sid = re.match(r"^(\d+)", base_name).group(1) if re.match(r"^(\d+)", base_name) else base_name.split(".")[0]

            ppg_raw, fs_orig, breath_idx = load_capnobase_h5py_pure(mat_path)
            if ppg_raw is None or len(ppg_raw) < 1000 or len(breath_idx) == 0:
                print(f"  [SKIP] Subjek {sid}: Gagal membaca data matriks.")
                continue

            if np.max(breath_idx) < 600.0:
                breath_idx = (breath_idx * fs_orig).astype(int)
            else:
                breath_idx = breath_idx.astype(int)

            ds_factor = max(1, int(round(fs_orig / fs_target)))
            ppg_ds = downsample(interpolate_nan(ppg_raw), ds_factor)
            if getattr(cfg.data, "use_bandpass_filter", True):
                ppg_ds = bandpass_filter(ppg_ds, fs_target, cfg.data.bandpass_low_hz, cfg.data.bandpass_high_hz, cfg.data.bandpass_order)
            if getattr(cfg.data, "use_zscore_normalization", True):
                ppg_ds = zscore_normalize(ppg_ds)

            win_orig = int(win_sec * fs_orig)
            step_orig = int(step_sec * fs_orig)
            n_win = (len(ppg_ds) - samples_per_win) // step_samples + 1
            sub_segs = 0

            for w in range(max(0, n_win)):
                st, et = w * step_samples, w * step_samples + samples_per_win
                if et > len(ppg_ds): break
                x = ppg_ds[st:et]

                so, eo = w * step_orig, w * step_orig + win_orig
                n_b = np.sum((breath_idx >= so) & (breath_idx < eo))
                rr_label = (n_b / win_sec) * 60.0

                if cfg.data.rr_min_bpm <= rr_label <= cfg.data.rr_max_bpm and np.std(x) > 1e-4:
                    segments.append({"x": x.astype(np.float32), "y": np.float32(rr_label), "subject_id": sid})
                    sub_segs += 1

            if sub_segs > 0:
                valid_count += 1
                print(f"  -> CapnoBase Subjek [{valid_count:02d}/{len(mat_files):02d}] (ID: {sid}): {sub_segs} segmen berhasil diekstrak.")

        except Exception as e:
            print(f"  [WARN] Subjek {mat_path}: {e}")

    if segments:
        x_t = torch.tensor(np.array([s["x"] for s in segments]), dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(np.array([s["y"] for s in segments]), dtype=torch.float32).view(-1, 1)
        subjs = [s["subject_id"] for s in segments]
        torch.save({"x": x_t, "y": y_t, "labels": y_t, "subject_ids": subjs}, save_path)
        print("==================================================================")
        print(f"[SUKSES] CapnoBase selesai: {len(x_t):,} segmen tersimpan di {save_path}")
        print(f"Rentang Nilai RR: {float(y_t.min()):.2f} - {float(y_t.max()):.2f} bpm")
        print("==================================================================\n")
    else:
        print("[PERINGATAN] Tidak ada segmen CapnoBase yang berhasil dibuat.")


def main():
    cfg = get_config() if get_config is not None else Config()
    process_bidmc_dataset(cfg)
    if os.path.exists(cfg.RAW_CAPNOBASE_DIR):
        process_capnobase_dataset(cfg)

if __name__ == "__main__":
    main()