"""Part 3 — Stress testing: SNR cliff and temperature drift."""

import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

from dataset import preprocess_spectra, DATA_DIR
from models import ResNet1D, TinyResNet1D
from utils import TARGET_MEAN, TARGET_STD

CKPT_DIR = REPO_ROOT / "checkpoints"
PLOT_DIR = Path(__file__).parent / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _load_models(device):
    fp32 = ResNet1D(base_ch=32)
    fp32.load_state_dict(torch.load(
        CKPT_DIR / "resnet_smooth_seed42_best.pt", map_location="cpu", weights_only=True))
    fp32.eval().to(device)

    student = TinyResNet1D()
    student.load_state_dict(torch.load(
        CKPT_DIR / "v2_student_best.pt", map_location="cpu", weights_only=True))
    student.eval().to(device)

    return fp32, student


@torch.no_grad()
def run_model_on_spectra(model, spectra_np, device, batch_size=256):
    """Preprocess raw spectra, run inference, return predictions in mT."""
    spec = preprocess_spectra(spectra_np)
    spec_t = torch.from_numpy(spec)
    preds = []
    for i in range(0, len(spec_t), batch_size):
        out = model(spec_t[i:i+batch_size].to(device)).cpu()
        preds.append(out)
    preds_z = torch.cat(preds).numpy()
    return preds_z * TARGET_STD + TARGET_MEAN


# ═════════════════════════════════════════════════════════════════════
# INVESTIGATION 6 — SNR Cliff
# ═════════════════════════════════════════════════════════════════════

def investigation6_snr_cliff(fp32, student, device):
    print("\n" + "=" * 60)
    print("INVESTIGATION 6: SNR Cliff")
    print("=" * 60)

    d = np.load(DATA_DIR / "stress_snr_sweep.npz")
    spectra = d["spectra"]
    b_mag = d["b_magnitude"]
    snr = d["snr"]
    b_idx = d["b_field_idx"]
    snr_levels = d["snr_levels"]
    b_fields = d["b_fields_mT"]

    models = [("FP32 Baseline", fp32), ("Distilled Student", student)]

    results = {}
    for name, model in models:
        preds = run_model_on_spectra(model, spectra, device)
        err_ut = np.abs(preds - b_mag) * 1000
        results[name] = err_ut

    # ── Group by (b_field, snr_level), compute mean MAE ──
    b_colors = ["tab:blue", "tab:orange", "tab:green"]
    b_labels = [f"B={b:.0f} mT" for b in b_fields]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    cliff_rows = []

    for ax, (name, _) in zip(axes, models):
        err_ut = results[name]
        for bi in range(len(b_fields)):
            mask_b = b_idx == bi
            mae_per_snr = []
            for snr_val in snr_levels:
                mask = mask_b & (np.abs(snr - snr_val) < 0.1)
                if mask.sum() > 0:
                    mae_per_snr.append(err_ut[mask].mean())
                else:
                    mae_per_snr.append(np.nan)
            mae_per_snr = np.array(mae_per_snr)
            ax.plot(snr_levels, mae_per_snr, "o-", color=b_colors[bi],
                    label=b_labels[bi], markersize=4, linewidth=1.5)

            valid = np.where(mae_per_snr < 1000)[0]
            cliff_snr = snr_levels[valid[0]] if len(valid) > 0 else float("inf")
            cliff_rows.append((name, b_fields[bi], cliff_snr))

            if np.isfinite(cliff_snr):
                ax.axvline(cliff_snr, color=b_colors[bi], linestyle=":",
                           linewidth=0.8, alpha=0.6)

        ax.axhline(1000, color="gray", linestyle="--", linewidth=1, label="MAE=1000 μT")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("SNR")
        ax.set_ylabel("MAE (μT)")
        ax.set_title(name)
        ax.legend(fontsize=8)

    fig.suptitle("SNR Cliff: MAE vs SNR by B-field", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stress_snr_cliff.png", dpi=150)
    plt.close(fig)
    print("  Saved stress_snr_cliff.png")

    # ── Cliff table ──
    print(f"\n  {'Model':25s} | {'B-field':>8s} | {'SNR cliff':>10s}")
    print(f"  {'-'*50}")
    for name, bf, cliff in cliff_rows:
        cliff_str = f"{cliff:.0f}" if np.isfinite(cliff) else "never"
        print(f"  {name:25s} | {bf:>6.0f} mT | {cliff_str:>10s}")

    # ── Sharpness analysis ──
    print(f"\n  Transition sharpness:")
    for name, _ in models:
        err_ut = results[name]
        for bi in range(len(b_fields)):
            mask_b = b_idx == bi
            cliff_snr = next(
                (c for n, bf, c in cliff_rows if n == name and bf == b_fields[bi]),
                float("inf"))
            if not np.isfinite(cliff_snr):
                continue
            cliff_idx = np.argmin(np.abs(snr_levels - cliff_snr))
            if cliff_idx > 0:
                mask_at = mask_b & (np.abs(snr - snr_levels[cliff_idx]) < 0.1)
                mask_below = mask_b & (np.abs(snr - snr_levels[cliff_idx - 1]) < 0.1)
                mae_at = err_ut[mask_at].mean() if mask_at.sum() > 0 else np.nan
                mae_below = err_ut[mask_below].mean() if mask_below.sum() > 0 else np.nan
                if not np.isnan(mae_at) and not np.isnan(mae_below):
                    ratio = mae_below / mae_at
                    sharp = "SHARP" if ratio > 3 else "gradual"
                    print(f"    {name}, B={b_fields[bi]:.0f}mT: "
                          f"MAE at cliff={mae_at:.0f}, one step below={mae_below:.0f}, "
                          f"ratio={ratio:.1f}x → {sharp}")

    # ── Compare models ──
    fp32_cliffs = {bf: c for n, bf, c in cliff_rows if n == "FP32 Baseline"}
    stud_cliffs = {bf: c for n, bf, c in cliff_rows if n == "Distilled Student"}
    same = all(fp32_cliffs.get(bf) == stud_cliffs.get(bf) for bf in b_fields)

    print(f"\n  SNR cliffs identical between FP32 and student: {same}")
    if not same:
        for bf in b_fields:
            fc = fp32_cliffs.get(bf, float("inf"))
            sc = stud_cliffs.get(bf, float("inf"))
            if fc != sc:
                print(f"    B={bf:.0f}mT: FP32 cliff={fc:.0f}, Student cliff={sc:.0f}")

    print(f"\n  Physical explanation:")
    print(f"    The SNR cliff exists because dip detectability requires a minimum")
    print(f"    signal-to-noise ratio per dip (snr_per_dip = contrast * snr).")
    print(f"    Below this threshold the dips drown in photon shot noise and the")
    print(f"    spectrum becomes featureless — the model has nothing to latch onto.")
    print(f"    At B=1mT the dips overlap heavily so even moderate noise destroys")
    print(f"    the merged feature. At B=9mT the dips are wide apart but shallow,")
    print(f"    requiring higher SNR to detect each individual dip.")


# ═════════════════════════════════════════════════════════════════════
# INVESTIGATION 7A — Temperature Drift
# ═════════════════════════════════════════════════════════════════════

def investigation7a_temp_drift(fp32, student, device):
    print("\n" + "=" * 60)
    print("INVESTIGATION 7A: Temperature Drift (D shift)")
    print("=" * 60)

    d = np.load(DATA_DIR / "stress_temp_drift.npz")
    spectra = d["spectra"]
    b_mag = d["b_magnitude"]
    d_shift = d["d_shift_mhz"]

    shifts = np.sort(np.unique(d_shift))
    models = [("FP32 Baseline", fp32), ("Distilled Student", student)]
    model_colors = ["tab:blue", "tab:green"]

    results = {}
    for name, model in models:
        preds = run_model_on_spectra(model, spectra, device)
        err_ut = np.abs(preds - b_mag) * 1000
        results[name] = err_ut

    # ── Print per-shift MAE ──
    mae_table = {}
    for name, _ in models:
        mae_table[name] = {}
        print(f"\n  {name}:")
        for s in shifts:
            mask = d_shift == s
            mae = results[name][mask].mean()
            mae_table[name][s] = mae
            print(f"    ΔD={s:+6.0f} MHz  →  MAE = {mae:.1f} μT")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 6))
    for (name, _), color in zip(models, model_colors):
        maes = [mae_table[name][s] for s in shifts]
        ax.plot(shifts, maes, "o-", color=color, label=name, linewidth=2, markersize=6)

    baseline_mae = mae_table["FP32 Baseline"][0.0]
    ax.axhline(baseline_mae, color="gray", linestyle="--", linewidth=0.8,
               label=f"Baseline at ΔD=0 ({baseline_mae:.0f} μT)")
    ax.axhline(1000, color="red", linestyle="--", linewidth=0.8, label="1000 μT threshold")

    for (name, _), color in zip(models, model_colors):
        maes_arr = np.array([mae_table[name][s] for s in shifts])
        crossings = shifts[maes_arr > 1000]
        if len(crossings) > 0:
            for c in [crossings.min(), crossings.max()]:
                ax.axvline(c, color=color, linestyle=":", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("D shift (MHz)")
    ax.set_ylabel("MAE (μT)")
    ax.set_title("MAE vs Zero-Field Splitting Shift (Temperature Drift)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stress_temp_drift.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved stress_temp_drift.png")

    # ── Double-baseline threshold ──
    double_threshold = baseline_mae * 2
    print(f"\n  Double-baseline threshold: {double_threshold:.0f} μT")
    for name, _ in models:
        exceeded = [s for s in shifts if mae_table[name][s] > double_threshold]
        if exceeded:
            first_pos = min([s for s in exceeded if s > 0], default=None)
            first_neg = max([s for s in exceeded if s < 0], default=None)
            print(f"    {name}: exceeds at ΔD = {exceeded}")
            if first_pos and first_neg:
                print(f"      First positive: +{first_pos:.0f} MHz, "
                      f"first negative: {first_neg:.0f} MHz")
        else:
            print(f"    {name}: never exceeds 2× baseline across tested shifts")

    # ── Symmetry check ──
    print(f"\n  Symmetry check:")
    for name, _ in models:
        for mag in [5, 10, 20]:
            pos = mae_table[name].get(float(mag))
            neg = mae_table[name].get(float(-mag))
            if pos is not None and neg is not None:
                ratio = max(pos, neg) / (min(pos, neg) + 1e-6)
                sym = "symmetric" if ratio < 1.3 else "asymmetric"
                print(f"    {name}: ΔD=±{mag} MHz  "
                      f"+{pos:.0f} / -{neg:.0f} μT  (ratio={ratio:.2f}, {sym})")

    # ── Physical explanation ──
    print(f"\n  Physical explanation:")
    print(f"    A shift in D moves all 8 resonance dips by the same amount.")
    print(f"    The model was trained with D=2.87 GHz. A drift of X MHz shifts")
    print(f"    all dips by X MHz. Our per-sample z-score normalization absorbs")
    print(f"    baseline intensity changes, and gaussian smoothing (sigma=4, ~4")
    print(f"    frequency bins ≈ 4.3 MHz) blurs small shifts. But beyond ~10 MHz")
    print(f"    the dips land at positions never seen during training, causing")
    print(f"    the model's learned spectral templates to mismatch.")

    print(f"\n  Proposed augmentation strategy:")
    print(f"    During training, randomly shift all dip positions by")
    print(f"    ΔD ~ Uniform(-20, +20) MHz by generating spectra with D sampled")
    print(f"    from N(2870, 10) MHz. This teaches the model that dip positions")
    print(f"    can vary systematically while B-field magnitude is still")
    print(f"    recoverable from relative dip spacings.")


# ═════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    fp32, student = _load_models(device)
    print("Models loaded: FP32 Baseline, Distilled Student")

    investigation6_snr_cliff(fp32, student, device)
    investigation7a_temp_drift(fp32, student, device)

    print("\n" + "=" * 60)
    print("All stress tests complete. Plots saved to part3/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
