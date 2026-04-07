"""
EDA script for NV-center ODMR magnetometer dataset.
Prints shapes, statistics, anomaly counts, and data-integrity checks.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ── helper ──────────────────────────────────────────────────────────────────
def check_nan_inf(name: str, arr: np.ndarray) -> list[str]:
    """Return a list of warning strings if NaN or Inf values are found."""
    warnings = []
    nan_count = np.count_nonzero(np.isnan(arr))
    inf_count = np.count_nonzero(np.isinf(arr))
    if nan_count:
        warnings.append(f"  ⚠  {name}: {nan_count} NaN values found")
    if inf_count:
        warnings.append(f"  ⚠  {name}: {inf_count} Inf values found")
    return warnings


def print_stat_line(name: str, arr: np.ndarray) -> None:
    """Print shape + basic statistics for a single array."""
    shape_str = str(arr.shape)
    print(
        f"  {name:20s}: shape {shape_str:20s}  "
        f"min {arr.min():12.4f}  max {arr.max():12.4f}  "
        f"mean {arr.mean():12.4f}  std {arr.std():12.4f}"
    )


def plot_spectra_grid(
    train_data: np.lib.npyio.NpzFile,
    freq_axis: np.ndarray,
    target_fields_mt: list[float] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
) -> None:
    """Plot vertically-stacked ODMR spectra at selected B-field magnitudes."""

    PLOTS_DIR = Path(__file__).parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    freq_ghz = freq_axis / 1e9
    b_mag = train_data["b_magnitude"]
    spectra = train_data["spectra"]

    # Find the single sample closest to each target field value
    indices = []
    for target in target_fields_mt:
        idx = int(np.argmin(np.abs(b_mag - target)))
        indices.append(idx)

    D_GHZ = 2.87  # zero-field splitting

    fig, axes = plt.subplots(
        len(target_fields_mt), 1,
        figsize=(10, 18),
        sharex=True,
    )

    for ax, idx, target in zip(axes, indices, target_fields_mt):
        ax.plot(freq_ghz, spectra[idx], color="#1f77b4", linewidth=0.8)
        ax.axvline(D_GHZ, color="gray", linestyle="--", linewidth=0.7)
        ax.set_ylabel("PL (a.u.)", fontsize=8)
        ax.set_title(f"B = {b_mag[idx]:.2f} mT", fontsize=9)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Frequency (GHz)", fontsize=9)
    fig.tight_layout()

    out_path = PLOTS_DIR / "spectra_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── 1. load data ────────────────────────────────────────────────────────────
splits = {}
for split in ("train", "val", "test"):
    path = DATA_DIR / f"{split}.npz"
    splits[split] = np.load(path)
    print(f"Loaded {path.name}")

freq_axis = np.load(DATA_DIR / "freq_axis.npy")
print(f"Loaded freq_axis.npy\n")

# ── 2. shapes of every array per split ──────────────────────────────────────
for split, data in splits.items():
    print(f"--- {split}.npz ---")
    for key in sorted(data.files):
        print(f"  {key:20s}: shape {str(data[key].shape)}")
    print()

# ── 3. detailed statistics (training set only) ─────────────────────────────
stat_keys = ["spectra", "b_magnitude", "snr", "linewidth", "contrast"]
print("--- Training-set statistics ---")
for key in stat_keys:
    if key in splits["train"].files:
        print_stat_line(key, splits["train"][key])
    else:
        print(f"  {key:20s}: *** NOT FOUND ***")
print()

# ── 4. anomaly counts ──────────────────────────────────────────────────────
print("--- Anomaly counts ---")
for split, data in splits.items():
    flags = data["is_anomalous"].astype(bool)
    total = len(flags)
    count = int(flags.sum())
    frac = count / total
    print(f"  {split:6s}: {count:6d} / {total:6d}  ({frac:.4f})")
print()

# ── 5. freq_axis info ──────────────────────────────────────────────────────
print("--- freq_axis.npy ---")
print(
    f"  {len(freq_axis)} points, "
    f"{freq_axis.min() / 1e9:.3f} GHz to {freq_axis.max() / 1e9:.3f} GHz"
)
print()

# ── 6. NaN / Inf check ─────────────────────────────────────────────────────
print("--- NaN / Inf check ---")
all_warnings: list[str] = []

for split, data in splits.items():
    for key in sorted(data.files):
        arr = data[key]
        if np.issubdtype(arr.dtype, np.floating):
            all_warnings.extend(check_nan_inf(f"{split}/{key}", arr))

if np.issubdtype(freq_axis.dtype, np.floating):
    all_warnings.extend(check_nan_inf("freq_axis", freq_axis))

if all_warnings:
    for w in all_warnings:
        print(w)
else:
    print("  ✓ No NaN or Inf values found in any array.")

# ── 7. spectra grid plot ────────────────────────────────────────────────────
plot_spectra_grid(splits["train"], freq_axis)

print("\nDone.")
