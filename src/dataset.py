import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import Config

class PPGDataset(Dataset):
    def __init__(self, x_data, y_data, subject_ids=None):
        self.x = torch.as_tensor(x_data, dtype=torch.float32)
        self.y = torch.as_tensor(y_data, dtype=torch.float32).view(-1, 1)
        self.subject_ids = subject_ids
        if self.x.ndim == 2:
            self.x = self.x.unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloaders(config=None, dataset_name=None, batch_size=None, num_workers=None, **kwargs):
    cfg = config if config is not None else Config()
    b_size = batch_size if batch_size is not None else getattr(cfg.train, "batch_size", 128)
    workers = num_workers if num_workers is not None else getattr(cfg.train, "num_workers", 0)
    d_name = dataset_name if dataset_name is not None else getattr(cfg, "ACTIVE_DATASET", "BIDMC")

    data_path = cfg.CAPNOBASE_PROCESSED_PATH if d_name.upper() == "CAPNOBASE" else cfg.BIDMC_PROCESSED_PATH

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset '{data_path}' belum ditemukan. Jalankan 'python -m src.preprocessing' terlebih dahulu.")

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    x_all = data["x"]
    y_all = data["y"]
    subj_all = np.array(data.get("subject_ids", [f"s_{i//20}" for i in range(len(x_all))]))

    # Subject-Wise Split sesuai konfigurasi asli
    unique_subjects = np.unique(subj_all)
    np.random.seed(cfg.split.random_seed)
    shuffled_subjs = np.random.permutation(unique_subjects)

    n_train = min(cfg.split.n_train_subjects, len(unique_subjects) - 2)
    n_val = min(cfg.split.n_val_subjects, max(1, (len(unique_subjects) - n_train) // 2))

    train_subjs = shuffled_subjs[:n_train]
    val_subjs = shuffled_subjs[n_train : n_train + n_val]
    test_subjs = shuffled_subjs[n_train + n_val :]

    train_idx = np.isin(subj_all, train_subjs)
    val_idx = np.isin(subj_all, val_subjs)
    test_idx = np.isin(subj_all, test_subjs)

    print(f"\n[SPLIT: {d_name.upper()}] Subjek -> Train:{len(train_subjs)} Val:{len(val_subjs)} Test:{len(test_subjs)}")
    print(f"[SPLIT: {d_name.upper()}] Segmen -> Train:{np.sum(train_idx):,} Val:{np.sum(val_idx):,} Test:{np.sum(test_idx):,}")

    train_ds = PPGDataset(x_all[train_idx], y_all[train_idx], subj_all[train_idx])
    val_ds = PPGDataset(x_all[val_idx], y_all[val_idx], subj_all[val_idx])
    test_ds = PPGDataset(x_all[test_idx], y_all[test_idx], subj_all[test_idx])

    return (
        DataLoader(train_ds, batch_size=b_size, shuffle=True, num_workers=workers),
        DataLoader(val_ds, batch_size=b_size, shuffle=False, num_workers=workers),
        DataLoader(test_ds, batch_size=b_size, shuffle=False, num_workers=workers)
    )

def get_rr_stratified_subsets(x_data, y_data):
    x_np = x_data.cpu().numpy() if isinstance(x_data, torch.Tensor) else np.array(x_data)
    y_np = y_data.cpu().numpy().flatten() if isinstance(y_data, torch.Tensor) else np.array(y_data).flatten()
    return {
        "Bradypnea (<12 bpm)": (x_np[y_np < 12.0], y_np[y_np < 12.0]),
        "Normal (12-20 bpm)": (x_np[(y_np >= 12.0) & (y_np <= 20.0)], y_np[(y_np >= 12.0) & (y_np <= 20.0)]),
        "Tachypnea (>20 bpm)": (x_np[y_np > 20.0], y_np[y_np > 20.0])
    }