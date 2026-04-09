"""
Part 1C: Interpretability — Saliency, Occlusion & Synthetic Probes
===================================================================
Run on Kaggle:  %run interpretability.py

Requires:
  - /kaggle/working/checkpoints/dilresnet_seed777.pt   (DilatedResNet1D weights)
  - /kaggle/working/checkpoints/norm_params.json
  - /kaggle/input/odmr-data/data/test.npz
  - /kaggle/input/odmr-data/data/freq_axis.npy
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR  = Path("/kaggle/input/datasets/krishivgupta123123/diraclab/data")
CKPT_DIR  = Path("/kaggle/working/checkpoints")
PLOT_DIR  = Path("/kaggle/working/plots")
CKPT_PATH = CKPT_DIR / "dilresnet_seed777.pt" #best model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_GHZ  = 2.87
GAMMA_NV = 28.024  # GHz / T


# ═══════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

with open(CKPT_DIR / "norm_params.json") as f:
    _norm = json.load(f)
SPEC_MEAN = _norm["mean"]
SPEC_STD  = _norm["std"]


def normalize(spectrum: np.ndarray) -> np.ndarray:
    """Z-score normalize using training-set mean / std."""
    return (spectrum - SPEC_MEAN) / SPEC_STD


# ═══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  (must match train_models.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

class _DilResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation),
        )

    def forward(self, x):
        return x + self.net(x)


class _DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DilatedResNet1D(nn.Module):
    def __init__(self, base_ch=64, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, 7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.Sequential(
            *[_DilResBlock(base_ch, d) for d in dilations]
        )
        self.down = nn.Sequential(
            _DownBlock(base_ch, base_ch * 2),
            _DownBlock(base_ch * 2, base_ch * 4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(base_ch * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dilated(x)
        x = self.down(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════════

print("Loading data …")
test_npz  = np.load(DATA_DIR / "test.npz")
freq_axis = np.load(DATA_DIR / "freq_axis.npy")
freq_ghz  = freq_axis / 1e9

test_spectra  = test_npz["spectra"]        # (5000, 512)  raw
test_b_mag    = test_npz["b_magnitude"]    # (5000,)  mT
test_snr      = test_npz["snr"]

print(f"  test spectra {test_spectra.shape}, freq axis {freq_ghz.shape}")
print(f"  norm  mean={SPEC_MEAN:.6f}  std={SPEC_STD:.6f}")

print("Loading model …")
model = DilatedResNet1D().to(DEVICE)
model.load_state_dict(
    torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()
print(f"  Loaded {CKPT_PATH.name} on {DEVICE}\n")


def predict_np(spectra_raw: np.ndarray) -> np.ndarray:
    """Run inference on raw (unnormalized) spectra.  Accepts (512,) or (N,512)."""
    if spectra_raw.ndim == 1:
        spectra_raw = spectra_raw[np.newaxis, :]
    normed = normalize(spectra_raw)
    t = torch.from_numpy(normed).float().unsqueeze(1).to(DEVICE)  # (N,1,512)
    with torch.no_grad():
        return model(t).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1 — GRADIENT SALIENCY
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 1: Gradient Saliency")
print("=" * 60)

PLOT_DIR.mkdir(parents=True, exist_ok=True)

bins = {
    "0-1":  (0.0,  1.0),
    "1-3":  (1.0,  3.0),
    "5-10": (5.0, 10.0),
}

for tag, (lo, hi) in bins.items():
    mask = (test_b_mag >= lo) & (test_b_mag < hi)
    idxs = np.where(mask)[0][:30]

    raw_batch  = test_spectra[idxs]                       # (<=30, 512)
    norm_batch = normalize(raw_batch)
    inp = torch.from_numpy(norm_batch).float().unsqueeze(1).to(DEVICE)  # (N,1,512)
    inp.requires_grad_(True)

    preds = model(inp)
    preds.sum().backward()
    saliency = inp.grad.abs().squeeze(1).cpu().numpy()    # (N, 512)

    mean_raw  = raw_batch.mean(axis=0)
    mean_sal  = saliency.mean(axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(freq_ghz, mean_raw, color="steelblue", linewidth=0.8, label="Mean spectrum")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("PL (raw)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.fill_between(freq_ghz, 0, mean_sal, color="orange", alpha=0.45, label="Mean saliency")
    ax2.set_ylabel("|∂pred/∂input|", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.axvline(D_GHZ, color="gray", linestyle="--", linewidth=0.7, label="D = 2.87 GHz")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.set_title(f"Gradient Saliency — |B| ∈ [{lo}, {hi}) mT  (n={len(idxs)})")
    fig.tight_layout()

    out = PLOT_DIR / f"saliency_bin_{tag.replace('-','_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")

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

records = []   # (sample_idx, dip_rank, abs_delta, distance_from_D)

for si in occl_idxs:
    raw  = test_spectra[si]
    norm_spec = normalize(raw)

    # detect dips — exact method specified in the task
    smoothed = savgol_filter(raw, 11, 3)
    inverted = -smoothed
    peaks, _ = find_peaks(inverted, prominence=0.03, distance=4)

    if len(peaks) == 0:
        continue

    # original prediction (on normalized spectrum)
    inp_orig = torch.from_numpy(norm_spec).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_orig = model(inp_orig).item()

    dip_freqs = freq_ghz[peaks]
    dists     = np.abs(dip_freqs - D_GHZ)
    rank_order = np.argsort(dists)            # 0 = most inner

    for rank, pi in enumerate(rank_order):
        masked = norm_spec.copy()
        center = peaks[pi]
        lo_idx = max(0, center - WINDOW // 2)
        hi_idx = min(512, center + WINDOW // 2)
        masked[lo_idx:hi_idx] = 0.0           # dataset mean in z-space

        inp_m = torch.from_numpy(masked).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
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
ax.set_ylabel("Mean |Δ prediction| (mT)")
ax.set_title("Occlusion Study — prediction shift when masking each dip rank")

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#e74c3c", label="Inner half"),
                    Patch(color="#2ecc71", label="Outer half")], fontsize=9)
fig.tight_layout()
out = PLOT_DIR / "occlusion.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"  Saved {out}")
print(f"  Processed {N_OCCL} samples, {len(records)} dip-mask events")

inner = df.loc[df["label"] == "Inner", "delta"]
outer = df.loc[df["label"] == "Outer", "delta"]
print(f"  Inner dips — mean |Δ|: {inner.mean():.4f} mT")
print(f"  Outer dips — mean |Δ|: {outer.mean():.4f} mT")


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
    """Generate a synthetic ODMR spectrum (baseline=1) with Lorentzian dips.

    freqs   : frequency axis in GHz
    centers : list of dip center frequencies in GHz
    lw_mhz  : full-width at half-maximum in MHz
    contrast : fractional dip depth (0-1)
    """
    hw = (lw_mhz / 1000) / 2          # half-width in GHz
    spectrum = np.ones_like(freqs)
    for c in centers:
        spectrum -= contrast * hw**2 / ((freqs - c)**2 + hw**2)
    return spectrum


# ── Probe 1: Flat ──
probe1 = np.ones(512)

# ── Probe 2: 2-dip (single NV axis at 5 mT) ──
shift = GAMMA_NV * 0.005           # 0.005 T = 5 mT projected fully on one axis
probe2 = make_lorentzian(freq_ghz, [D_GHZ - shift, D_GHZ + shift], lw_mhz=10, contrast=0.03)

# ── Probe 3: Impossible (dips outside normal range) ──
probe3 = make_lorentzian(freq_ghz, [2.50, 3.20], lw_mhz=10, contrast=0.03)

# ── Probe 4: 16-dip (real dips + ghost dips shifted +5 MHz) ──
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
    "Probe 1 (Flat baseline)":         probe1,
    "Probe 2 (2-dip, 5 mT single NV)": probe2,
    "Probe 3 (Impossible dips)":       probe3,
    f"Probe 4 (16-dip, real+ghost, {len(real_centers)}+{len(ghost_centers)} dips)": probe4,
}

print()
for name, spec in probes.items():
    pred = predict_np(spec)
    print(f"  {name:50s}  ->  {pred[0]:.4f} mT")

print("\n" + "=" * 60)
print("Interpretability analysis complete.")
print("=" * 60)
