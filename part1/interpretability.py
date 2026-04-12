"""
Part 1C: Interpretability — Saliency, Occlusion & Synthetic Probes
===================================================================
Uses ResNet1D(base_ch=32) from shared/models.py with the standard
preprocessing pipeline (invert -> smooth sigma=4 -> per-sample zscore).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter, find_peaks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

from dataset import preprocess_spectra
from models import ResNet1D

DATA_DIR  = REPO_ROOT / "data"
CKPT_DIR  = REPO_ROOT / "checkpoints"
PLOT_DIR  = Path(__file__).parent / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = CKPT_DIR / "resnet_smooth_seed42_best.pt"

TARGET_MEAN = 4.8513
TARGET_STD  = 2.3309

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
D_GHZ  = 2.87
GAMMA_NV = 28.024  # GHz / T


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════════

print("Loading data ...")
test_npz  = np.load(DATA_DIR / "test.npz")
freq_axis = np.load(DATA_DIR / "freq_axis.npy")
freq_ghz  = freq_axis / 1e9

test_spectra  = test_npz["spectra"]        # (5000, 512)  raw
test_b_mag    = test_npz["b_magnitude"]    # (5000,)  mT
test_snr      = test_npz["snr"]

print(f"  test spectra {test_spectra.shape}, freq axis {freq_ghz.shape}")

print("Loading model ...")
model = ResNet1D(base_ch=32).to(DEVICE)
model.load_state_dict(
    torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()
print(f"  Loaded {CKPT_PATH.name} on {DEVICE}\n")


def predict_np(spectra_raw: np.ndarray) -> np.ndarray:
    """Run inference on raw spectra using the standard preprocessing pipeline.
    Accepts (512,) or (N,512). Returns predictions in mT."""
    if spectra_raw.ndim == 1:
        spectra_raw = spectra_raw[np.newaxis, :]
    preprocessed = preprocess_spectra(spectra_raw, sigma=4.0)
    t = torch.from_numpy(preprocessed).float().to(DEVICE)
    with torch.no_grad():
        out_z = model(t).cpu().numpy()
    return out_z * TARGET_STD + TARGET_MEAN


def preprocess_for_grad(spectra_raw: np.ndarray) -> np.ndarray:
    """Preprocess raw spectra, return float32 array ready for model input."""
    if spectra_raw.ndim == 1:
        spectra_raw = spectra_raw[np.newaxis, :]
    return preprocess_spectra(spectra_raw, sigma=4.0)


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1 — GRADIENT SALIENCY
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 1: Gradient Saliency")
print("=" * 60)

bins = {
    "0-1":  (0.0,  1.0),
    "1-3":  (1.0,  3.0),
    "5-10": (5.0, 10.0),
}

for tag, (lo, hi) in bins.items():
    mask = (test_b_mag >= lo) & (test_b_mag < hi)
    idxs = np.where(mask)[0][:30]

    raw_batch  = test_spectra[idxs]
    proc_batch = preprocess_for_grad(raw_batch)
    inp = torch.from_numpy(proc_batch).float().to(DEVICE)
    inp.requires_grad_(True)

    preds = model(inp)
    preds.sum().backward()
    saliency = inp.grad.abs().cpu().numpy()    # (N, 512)

    mean_raw  = raw_batch.mean(axis=0)
    mean_sal  = saliency.mean(axis=0)

    peak_idx = np.argmax(mean_sal)
    peak_freq = freq_ghz[peak_idx]
    print(f"  Bin [{lo}, {hi}) mT: peak saliency at {peak_freq:.4f} GHz "
          f"(idx {peak_idx}), mean sal = {mean_sal.mean():.6f}, "
          f"max sal = {mean_sal.max():.6f}")

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(freq_ghz, mean_raw, color="steelblue", linewidth=0.8, label="Mean spectrum")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("PL (raw)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.fill_between(freq_ghz, 0, mean_sal, color="orange", alpha=0.45, label="Mean saliency")
    ax2.set_ylabel("|d(pred)/d(input)|", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.axvline(D_GHZ, color="gray", linestyle="--", linewidth=0.7, label="D = 2.87 GHz")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.set_title(f"Gradient Saliency -- |B| in [{lo}, {hi}) mT  (n={len(idxs)})")
    fig.tight_layout()

    out = PLOT_DIR / f"saliency_bin_{tag.replace('-','_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    Saved {out.name}")

    model.zero_grad()


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2 — OCCLUSION STUDY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 2: Occlusion Study")
print("=" * 60)

N_OCCL   = 100
WINDOW   = 20

rng = np.random.default_rng(42)
occl_idxs = rng.choice(len(test_spectra), size=N_OCCL, replace=False)

records = []

for si in occl_idxs:
    raw  = test_spectra[si]

    smoothed = savgol_filter(raw, 11, 3)
    inverted = -smoothed
    peaks, _ = find_peaks(inverted, prominence=0.03, distance=4)

    if len(peaks) == 0:
        continue

    proc_spec = preprocess_for_grad(raw)[0]  # (512,)

    inp_orig = torch.from_numpy(proc_spec).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_orig = model(inp_orig).item()

    dip_freqs = freq_ghz[peaks]
    dists     = np.abs(dip_freqs - D_GHZ)
    rank_order = np.argsort(dists)

    for rank, pi in enumerate(rank_order):
        masked = proc_spec.copy()
        center = peaks[pi]
        lo_idx = max(0, center - WINDOW // 2)
        hi_idx = min(512, center + WINDOW // 2)
        masked[lo_idx:hi_idx] = 0.0

        inp_m = torch.from_numpy(masked).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_masked = model(inp_m).item()

        records.append({
            "rank": rank,
            "delta": abs(pred_masked - pred_orig),
            "dist": dists[pi],
            "label": "Inner" if rank < len(peaks) // 2 else "Outer",
        })

import pandas as pd
df = pd.DataFrame(records)

max_rank = int(df["rank"].max()) + 1
rank_labels = [f"Rank {r}" for r in range(max_rank)]
rank_means  = [df.loc[df["rank"] == r, "delta"].mean() for r in range(max_rank)]
rank_colors = ["#e74c3c" if r < max_rank // 2 else "#2ecc71" for r in range(max_rank)]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(max_rank), rank_means, color=rank_colors, edgecolor="white")
ax.set_xticks(range(max_rank))
ax.set_xticklabels(rank_labels, fontsize=8)
ax.set_xlabel("Dip rank (0 = closest to D, higher = further)")
ax.set_ylabel("Mean |delta prediction| (mT)")
ax.set_title("Occlusion Study -- prediction shift when masking each dip rank")

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#e74c3c", label="Inner half"),
                    Patch(color="#2ecc71", label="Outer half")], fontsize=9)
fig.tight_layout()
out = PLOT_DIR / "occlusion.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"  Saved {out.name}")
print(f"  Processed {N_OCCL} samples, {len(records)} dip-mask events")

inner = df.loc[df["label"] == "Inner", "delta"]
outer = df.loc[df["label"] == "Outer", "delta"]
print(f"  Inner dips -- mean |delta|: {inner.mean():.4f} mT")
print(f"  Outer dips -- mean |delta|: {outer.mean():.4f} mT")
print(f"  Per-rank means: {[f'{m:.4f}' for m in rank_means]}")


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3 — SYNTHETIC PROBES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 3: Synthetic Probes")
print("=" * 60)


def make_lorentzian(
    freqs: np.ndarray,
    centers: list[float],
    lw_mhz: float,
    contrast: float,
) -> np.ndarray:
    hw = (lw_mhz / 1000) / 2
    spectrum = np.ones_like(freqs)
    for c in centers:
        spectrum -= contrast * hw**2 / ((freqs - c)**2 + hw**2)
    return spectrum


probe1 = np.ones(512)

shift = GAMMA_NV * 0.005
probe2 = make_lorentzian(freq_ghz, [D_GHZ - shift, D_GHZ + shift], lw_mhz=10, contrast=0.03)

probe3 = make_lorentzian(freq_ghz, [2.50, 3.20], lw_mhz=10, contrast=0.03)

target_5mT = np.argmin(np.abs(test_b_mag - 5.0))
raw_5mT = test_spectra[target_5mT]
smoothed_5 = savgol_filter(raw_5mT, 11, 3)
peaks_5, _ = find_peaks(-smoothed_5, prominence=0.03, distance=4)
real_centers = freq_ghz[peaks_5].tolist()
ghost_shift  = 0.005  # +5 MHz in GHz
ghost_centers = [c + ghost_shift for c in real_centers]
probe4 = make_lorentzian(
    freq_ghz,
    real_centers + ghost_centers,
    lw_mhz=10,
    contrast=0.03,
)

probes = {
    "Probe 1 (Flat baseline)":         (probe1, None),
    "Probe 2 (2-dip, 5 mT single NV)": (probe2, 5.0),
    "Probe 3 (Impossible dips)":       (probe3, None),
    f"Probe 4 (16-dip, real+ghost, {len(real_centers)}+{len(ghost_centers)} dips)": (probe4, 5.0),
}

print()
for name, (spec, true_b) in probes.items():
    pred = predict_np(spec)
    true_str = f"  (true={true_b:.1f} mT)" if true_b else ""
    print(f"  {name:55s}  ->  {pred[0]:.4f} mT{true_str}")

print("\n" + "=" * 60)
print("Interpretability analysis complete.")
print("=" * 60)
