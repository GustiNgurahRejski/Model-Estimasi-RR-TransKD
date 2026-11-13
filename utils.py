"""
utils.py
========
Fungsi bantu lintas-file: reproducibility, device selection, checkpointing,
dan CSV logger sederhana untuk mencatat hasil tiap eksperimen.
"""

import os
import csv
import random
import time
from typing import Optional, Dict, Any

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """Auto-fallback ke CPU kalau CUDA tidak tersedia."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: torch.nn.Module, path: str, extra: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return payload


class EarlyStopper:
    """Berhenti training kalau validation loss tidak membaik selama `patience` epoch."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CSVLogger:
    """Logger sederhana: setiap baris = satu event (epoch, hasil eval, dsb)."""

    def __init__(self, path: str, fieldnames: list):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        write_header = not os.path.exists(path)
        self._f = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=fieldnames)
        if write_header:
            self._writer.writeheader()
            self._f.flush()

    def log(self, row: Dict[str, Any]):
        self._writer.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


class Timer:
    """Context manager kecil untuk mengukur durasi suatu blok kode (ms)."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
