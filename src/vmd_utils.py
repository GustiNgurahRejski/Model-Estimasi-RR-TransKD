"""
src/vmd_utils.py
================
Modul VMD (Variational Mode Decomposition) sesuai metodologi TransRR (Zhao et al., 2023):
- Mengekstrak K = 5 mode intrinsik.
- Mengeliminasi mode ke-5 (Motion Artifacts).
- Merekonstruksi sinyal respirasi dari Mode 1 sampai Mode 4.
"""

import time
import numpy as np
import torch

try:
    from vmdpy import VMD
    HAS_VMDPY = True
except ImportError:
    HAS_VMDPY = False


def apply_vmd_1d(signal_1d: np.ndarray, K: int = 5, alpha: float = 2000.0, tau: float = 0.0, tol: float = 1e-7) -> np.ndarray:
    if not HAS_VMDPY:
        return signal_1d

    try:
        u, _, _ = VMD(signal_1d, alpha, tau, K, DC=0, init=1, tol=tol)
        reconstructed = np.sum(u[:4, :], axis=0)
        return reconstructed.astype(np.float32)
    except Exception:
        return signal_1d.astype(np.float32)


def apply_vmd_batch(x_tensor: torch.Tensor, K: int = 5, alpha: float = 2000.0) -> tuple[torch.Tensor, float]:
    """
    Menerapkan dekomposisi VMD pada batch [Batch_Size, 1, Seq_Len]
    dan mengembalikan tensor hasil rekonstruksi beserta estimasi latency per-sampel (ms).
    """
    b, _, _ = x_tensor.shape
    x_np = x_tensor.cpu().numpy()
    out_np = np.zeros_like(x_np)

    start_t = time.perf_counter()
    for i in range(b):
        out_np[i, 0, :] = apply_vmd_1d(x_np[i, 0, :], K=K, alpha=alpha)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0 / max(1, b)

    return torch.tensor(out_np, dtype=x_tensor.dtype, device=x_tensor.device), elapsed_ms