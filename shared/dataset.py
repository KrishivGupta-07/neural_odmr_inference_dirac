"""Preprocessing pipeline and data loading for ODMR spectra."""

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"


def preprocess_spectra(spectra_np: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Invert, smooth, per-sample z-score. Returns float32 array (N, 512)."""
    s = spectra_np.astype(np.float64)
    s = -(s - s.max(axis=1, keepdims=True))
    s = gaussian_filter1d(s, sigma=sigma, axis=1)
    mu = s.mean(axis=1, keepdims=True)
    sd = s.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    s = (s - mu) / sd
    return s.astype(np.float32)


def load_filtered(split: str, min_snr: float = 5.0, sigma: float = 4.0):
    """Load a split, filter by contrast*snr >= min_snr, preprocess.

    Returns (spectra_tensor (N,512), targets_tensor_mT (N,)).
    """
    npz = np.load(DATA_DIR / f"{split}.npz")
    spectra = npz["spectra"]
    b_mag = npz["b_magnitude"]
    contrast = npz["contrast"]
    snr = npz["snr"]

    mask = (contrast * snr) >= min_snr
    spectra = spectra[mask]
    b_mag = b_mag[mask]

    spectra = preprocess_spectra(spectra, sigma=sigma)
    return (
        torch.from_numpy(spectra),
        torch.from_numpy(b_mag.astype(np.float32)),
    )
