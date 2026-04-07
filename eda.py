"""
EDA script for NV-center ODMR magnetometer dataset.
Prints shapes, statistics, anomaly counts, and data-integrity checks.
Generates annotated spectra, dip-position analysis, distribution plots,
and overlap-regime investigation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

NV_AXES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
GAMMA_NV = 28.024  # GHz/T
D_GHZ = 2.87
NV_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
NV_LABELS = ["NV[111]", "NV[1-1-1]", "NV[-11-1]", "NV[-1-11]"]

# ── physics helpers ─────────────────────────────────────────────────────────

def theoretical_dip_positions(b_vec: np.ndarray) -> list[tuple[float, int]]:
    """Return sorted list of (freq_GHz, axis_index) for all 8 dips.

    b_vec must be in Tesla (matching the dataset's b_vector arrays).
    """
    dips = []
    for i, axis in enumerate(NV_AXES):
        proj = abs(np.dot(b_vec, axis))
        dips.append((D_GHZ - GAMMA_NV * proj, i))
        dips.append((D_GHZ + GAMMA_NV * proj, i))
    return sorted(dips, key=lambda x: x[0])


def extract_dips(
    spectrum: np.ndarray,
    freq_ghz: np.ndarray,
    distance: int = 4,
    smooth_window: int = 11,
) -> np.ndarray:
    
    # Smooth the noise, invert to find peaks
    smoothed = savgol_filter(spectrum, window_length=smooth_window, polyorder=3)
    inverted = -smoothed
    
    # Use the hardcoded threshold we proved works, with a small distance to catch overlaps
    peaks, _ = find_peaks(inverted, prominence=0.03, distance=distance)
    
    return freq_ghz[peaks]


# ── data-integrity helpers ─────────────────────────────────────────────────
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
    """Plot basic (unannotated) spectra grid -- kept for backward compat."""
    freq_ghz = freq_axis / 1e9
    b_mag = train_data["b_magnitude"]
    spectra = train_data["spectra"]

    indices = [int(np.argmin(np.abs(b_mag - t))) for t in target_fields_mt]

    fig, axes = plt.subplots(len(target_fields_mt), 1, figsize=(10, 18), sharex=True)
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


# ── Task 1: Annotated spectra with dip markers & theoretical positions ─────

def plot_annotated_spectra(
    train_data: np.lib.npyio.NpzFile,
    freq_axis: np.ndarray,
    target_fields_mt: list[float] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
) -> None:
    """Plot spectra annotated with detected dips (triangles) and
    theoretical NV-axis dip positions (colored dashed lines)."""
    freq_ghz = freq_axis / 1e9
    b_mag = train_data["b_magnitude"]
    spectra = train_data["spectra"]
    b_vecs = train_data["b_vector"]
    snr_arr = train_data["snr"]

    indices = [int(np.argmin(np.abs(b_mag - t))) for t in target_fields_mt]

    fig, axes = plt.subplots(len(target_fields_mt), 1, figsize=(12, 22), sharex=True)

    for ax, idx in zip(axes, indices):
        spec = spectra[idx]
        bv = b_vecs[idx]
        snr_val = snr_arr[idx]

        ax.plot(freq_ghz, spec, color="#1f77b4", linewidth=0.6, alpha=0.8)
        ax.axvline(D_GHZ, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

        # Theoretical dip positions from b_vector
        theo_dips = theoretical_dip_positions(bv)
        for freq_theo, axis_idx in theo_dips:
            if freq_ghz[0] <= freq_theo <= freq_ghz[-1]:
                ax.axvline(
                    freq_theo, color=NV_COLORS[axis_idx],
                    linestyle="--", linewidth=0.9, alpha=0.7,
                )

        detected = extract_dips(spec, freq_ghz)
        for df in detected:
            y_val = np.interp(df, freq_ghz, spec)
            ax.plot(df, y_val, "kv", markersize=5, alpha=0.8)

        ax.set_ylabel("PL", fontsize=8)
        ax.set_title(
            f"|B| = {b_mag[idx]:.2f} mT   SNR = {snr_val:.0f}   "
            f"detected {len(detected)} dips",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Frequency (GHz)", fontsize=9)

    legend_elements = [
        Line2D([0], [0], color=c, linestyle="--", label=l)
        for c, l in zip(NV_COLORS, NV_LABELS)
    ] + [Line2D([0], [0], marker="v", color="k", linestyle="None", label="Detected dip")]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = PLOTS_DIR / "spectra_annotated.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Task 2: Dip positions vs |B| scatter with theoretical overlay ──────────

def plot_dip_vs_B(
    train_data: np.lib.npyio.NpzFile,
    freq_axis: np.ndarray,
    n_samples: int = 1000,
) -> None:
    """Scatter-plot detected dip positions vs B-magnitude."""
    freq_ghz = freq_axis / 1e9
    b_mag = train_data["b_magnitude"]
    spectra = train_data["spectra"]

    rng = np.random.default_rng(42)
    selected = rng.choice(len(b_mag), size=min(n_samples, len(b_mag)), replace=False)

    det_b, det_f = [], []
    for idx in selected:
        dips = extract_dips(spectra[idx], freq_ghz)
        for d in dips:
            det_b.append(b_mag[idx])
            det_f.append(d)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(det_b, det_f, s=1.5, alpha=0.4, color="steelblue")
    ax.axhline(D_GHZ, color="gray", linestyle=":", linewidth=0.8, label=f"D = {D_GHZ} GHz")
    
    ax.set_xlabel("|B| (mT)")
    ax.set_ylabel("Dip frequency (GHz)")
    ax.set_title("ODMR Dip Positions vs Magnetic Field Magnitude")
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = PLOTS_DIR / "dip_splitting.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    out_path = PLOTS_DIR / "dip_positions_vs_B.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    print("\n--- Dip-position analysis ---")
    print("  Splitting pattern: Each NV axis projects B differently onto its symmetry")
    print("  direction. The Zeeman shift is f = D ± γ_NV·|B·n̂|, so the axis most")
    print("  aligned with B fans out fastest (largest |cos θ|). Near B=0 all 8 dips")
    print("  collapse to 2 (the ms=±1 degeneracy is barely lifted for any axis).")
    print(f"  Samples plotted: {len(selected)}, detected dip points: {len(det_b)}")


# ── Task 3: Distribution plots & hard-regime identification ────────────────

def plot_distributions(train_data: np.lib.npyio.NpzFile) -> None:
    """Plot SNR, linewidth, contrast distributions and identify the hard regime."""
    snr = train_data["snr"]
    lw = train_data["linewidth"]
    contrast = train_data["contrast"]
    b_mag = train_data["b_magnitude"]
    N = len(snr)

    # --- 3-panel histogram ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(snr, bins=60, color="#1f77b4", edgecolor="white", linewidth=0.3)
    axes[0].axvline(50, color="red", linestyle="--", linewidth=1, label="Hard threshold (50)")
    axes[0].set_xlabel("SNR")
    axes[0].set_ylabel("Count")
    axes[0].set_title("SNR Distribution")
    axes[0].legend(fontsize=8)

    axes[1].hist(lw, bins=60, color="#ff7f0e", edgecolor="white", linewidth=0.3)
    axes[1].axvline(10, color="red", linestyle="--", linewidth=1, label="Hard threshold (10 MHz)")
    axes[1].set_xlabel("Linewidth (MHz)")
    axes[1].set_title("Linewidth Distribution")
    axes[1].legend(fontsize=8)

    axes[2].hist(contrast, bins=60, color="#2ca02c", edgecolor="white", linewidth=0.3)
    axes[2].axvline(0.02, color="red", linestyle="--", linewidth=1, label="Hard threshold (0.02)")
    axes[2].set_xlabel("Contrast (fractional)")
    axes[2].set_title("Contrast Distribution")
    axes[2].legend(fontsize=8)

    fig.suptitle("Training Set Parameter Distributions", fontsize=13)
    fig.tight_layout()
    out_path = PLOTS_DIR / "distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # --- 2D correlation scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc = axes[0].scatter(snr, lw, c=contrast, s=1.5, alpha=0.4, cmap="viridis", vmin=0.01, vmax=0.07)
    axes[0].set_xlabel("SNR")
    axes[0].set_ylabel("Linewidth (MHz)")
    axes[0].set_title("SNR vs Linewidth (colored by Contrast)")
    plt.colorbar(sc, ax=axes[0], label="Contrast")

    sc2 = axes[1].hexbin(b_mag, lw, gridsize=40, cmap="inferno", mincnt=1)
    axes[1].set_xlabel("|B| (mT)")
    axes[1].set_ylabel("Linewidth (MHz)")
    axes[1].set_title("Linewidth vs |B| (density)")
    plt.colorbar(sc2, ax=axes[1], label="Count")

    fig.tight_layout()
    out_path = PLOTS_DIR / "hard_regime.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # --- Hard-regime statistics ---
    hard_snr = snr < 50
    hard_lw = lw > 10
    hard_con = contrast < 0.02
    hard_all = hard_snr & hard_lw & hard_con

    print("\n--- Hard-regime analysis ---")
    print(f"  Total training samples: {N}")
    print(f"  SNR < 50:              {hard_snr.sum():6d}  ({hard_snr.mean():.3%})")
    print(f"  Linewidth > 10 MHz:    {hard_lw.sum():6d}  ({hard_lw.mean():.3%})")
    print(f"  Contrast < 0.02:       {hard_con.sum():6d}  ({hard_con.mean():.3%})")
    print(f"  ALL three (hard):      {hard_all.sum():6d}  ({hard_all.mean():.3%})")

    if hard_all.sum() > 0:
        hard_b = b_mag[hard_all]
        print(f"  Hard-regime |B| range: {hard_b.min():.2f} – {hard_b.max():.2f} mT")
        print(f"  Hard-regime |B| mean:  {hard_b.mean():.2f} mT")


# ── Task 4 / Investigation 1: The Overlap Problem ─────────────────────────

def analyze_overlap(train_data: np.lib.npyio.NpzFile) -> None:
    """For each sample, compute 8 theoretical dip positions (using the
    already-validated theoretical_dip_positions helper, which takes b_vec
    in Tesla and uses GAMMA_NV = 28.024 GHz/T).  Flag as overlapping if
    the minimum adjacent-dip separation < the sample's linewidth."""
    b_mag = train_data["b_magnitude"]
    b_vecs = train_data["b_vector"]
    lw = train_data["linewidth"]
    N = len(b_mag)

    min_gaps_mhz = np.zeros(N)
    for i in range(N):
        dips = theoretical_dip_positions(b_vecs[i])  # b_vec already in Tesla
        freqs = np.array([f for f, _ in dips])
        min_gaps_mhz[i] = np.diff(freqs).min() * 1000  # GHz -> MHz

    overlapping = min_gaps_mhz < lw
    overlapping_b = b_mag[overlapping]
    resolved_b = b_mag[~overlapping]

    # ── stacked histogram ──
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 10, 50)
    ax.hist(overlapping_b, bins=bins, alpha=0.65, color="#e74c3c", label="Overlapping")
    ax.hist(resolved_b,    bins=bins, alpha=0.65, color="#2ecc71", label="Resolved")
    ax.set_xlabel("|B| (mT)")
    ax.set_ylabel("Count")
    ax.set_title("Overlap regime: ground-truth dip separation < linewidth")
    ax.legend()
    fig.tight_layout()
    out_path = PLOTS_DIR / "overlap_regime.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # ── printed summary ──
    overlap_pct = overlapping.mean() * 100
    print(f"\n--- Overlap analysis (Investigation 1) ---")
    print(f"  Total training samples: {N}")
    print(f"  Overlapping (min gap < linewidth): {overlapping.sum()} ({overlap_pct:.1f}%)")
    print(f"  Resolved: {(~overlapping).sum()} ({100 - overlap_pct:.1f}%)")
    if overlapping.sum() > 0:
        print(f"  Overlap-regime |B| range: {overlapping_b.min():.2f} – {overlapping_b.max():.2f} mT")
        print(f"  Overlap-regime |B| mean:  {overlapping_b.mean():.2f} mT")
    print()
    print("  Hypothesis: Models will exhibit higher prediction error in the overlap")
    print("  regime because adjacent dips from different NV axes merge into broad,")
    print("  ambiguous features, making the inverse mapping harder to learn.")
# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

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

# ── 7. spectra grid plot (basic) ──────────────────────────────────────────
plot_spectra_grid(splits["train"], freq_axis)

# ── 8. Task 1: Annotated spectra ──────────────────────────────────────────
print("\n" + "=" * 70)
print("TASK 1: Annotated Spectra Grid")
print("=" * 70)
plot_annotated_spectra(splits["train"], freq_axis)

# ── 9. Task 2: Dip positions vs |B| ──────────────────────────────────────
print("\n" + "=" * 70)
print("TASK 2: Dip Positions vs |B|")
print("=" * 70)
plot_dip_vs_B(splits["train"], freq_axis)

# ── 10. Task 3: Distributions & hard regime ───────────────────────────────
print("\n" + "=" * 70)
print("TASK 3: Parameter Distributions & Hard Regime")
print("=" * 70)
plot_distributions(splits["train"])

# ── 11. Task 4 / Investigation 1: Overlap analysis ───────────────────────
print("\n" + "=" * 70)
print("TASK 4 / INVESTIGATION 1: The Overlap Problem")
print("=" * 70)
analyze_overlap(splits["train"])

print("\n" + "=" * 70)
print("All EDA tasks complete.")
print("=" * 70)
