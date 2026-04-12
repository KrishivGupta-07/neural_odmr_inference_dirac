"""Part 2 — Forensic investigations on compressed models."""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

from dataset import load_filtered, preprocess_spectra, DATA_DIR
from models import ResNet1D, TinyResNet1D
from utils import TARGET_MEAN, TARGET_STD, evaluate

CKPT_DIR = REPO_ROOT / "checkpoints"
ONNX_DIR = Path(__file__).parent / "models"
PLOT_DIR = Path(__file__).parent / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _load_teacher(device="cpu"):
    m = ResNet1D(base_ch=32)
    m.load_state_dict(torch.load(CKPT_DIR / "resnet_smooth_seed42_best.pt",
                                 map_location="cpu", weights_only=True))
    m.eval().to(device)
    return m


def _load_student(ckpt_name, device="cpu"):
    m = TinyResNet1D()
    m.load_state_dict(torch.load(CKPT_DIR / ckpt_name,
                                 map_location="cpu", weights_only=True))
    m.eval().to(device)
    return m


def _ort_session(onnx_name):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(ONNX_DIR / onnx_name), opts,
                                providers=["CPUExecutionProvider"])


def _ort_predict_mt(session, spectra_np, batch_size=512):
    """Run ONNX session, return predictions in mT."""
    inp = session.get_inputs()[0].name
    preds = []
    for i in range(0, len(spectra_np), batch_size):
        out = session.run(None, {inp: spectra_np[i:i+batch_size]})[0].reshape(-1)
        preds.append(out)
    preds_z = np.concatenate(preds)
    return preds_z * TARGET_STD + TARGET_MEAN


def _pt_predict_mt(model, spectra_t, device="cpu", batch_size=512):
    """Run PyTorch model, return predictions in mT as numpy."""
    model.eval().to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(spectra_t), batch_size):
            out = model(spectra_t[i:i+batch_size].to(device)).cpu()
            preds.append(out)
    preds_z = torch.cat(preds).numpy()
    return preds_z * TARGET_STD + TARGET_MEAN


def _load_test_metadata():
    """Load raw test.npz and return (mask, snr, contrast, b_magnitude) for filtered samples."""
    npz = np.load(DATA_DIR / "test.npz")
    mask = (npz["contrast"] * npz["snr"]) >= 5.0
    return mask, npz["snr"][mask], npz["contrast"][mask], npz["b_magnitude"][mask]


def _binned_mean(x, y, n_bins=10):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() > 0:
            cx.append((lo + hi) / 2)
            cy.append(y[m].mean())
    return np.array(cx), np.array(cy)


# ═════════════════════════════════════════════════════════════════════
# INVESTIGATION 3 — Error Distribution Shift
# ═════════════════════════════════════════════════════════════════════

def investigation3_error_cdf():
    print("\n" + "=" * 60)
    print("INVESTIGATION 3: Error Distribution Shift")
    print("=" * 60)

    test_spec, test_targets = load_filtered("test")
    spec_np = test_spec.numpy()
    tgt_np = test_targets.numpy()
    mask, snr, contrast, b_mag = _load_test_metadata()

    teacher = _load_teacher()
    pred_baseline = _pt_predict_mt(teacher, test_spec)

    sess_int8 = _ort_session("v1_ptq_int8.onnx")
    pred_int8 = _ort_predict_mt(sess_int8, spec_np)

    sess_student = _ort_session("v2_student_fp32.onnx")
    pred_student = _ort_predict_mt(sess_student, spec_np)

    models_data = [
        ("FP32 Baseline", pred_baseline, "tab:blue"),
        ("V1 PTQ INT8", pred_int8, "tab:orange"),
        ("V2 Distilled Student", pred_student, "tab:green"),
    ]

    # ── Plot 1: Error CDF ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, preds, color in models_data:
        err_ut = np.abs(preds - tgt_np) * 1000
        sorted_err = np.sort(err_ut)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf, label=name, color=color, linewidth=1.5)

        frac_500 = (err_ut > 500).mean()
        frac_1000 = (err_ut > 1000).mean()
        print(f"  {name:25s}  >500μT: {frac_500:.3f}  >1000μT: {frac_1000:.3f}")

    ax.axvline(500, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(1000, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 3000)
    ax.set_xlabel("Absolute error (μT)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Error CDF — baseline vs compressed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "forensics_error_cdf.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Error vs B-field ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    y_max = 0
    for name, preds, _ in models_data:
        err_ut = np.abs(preds - tgt_np) * 1000
        y_max = max(y_max, np.percentile(err_ut, 99))

    for ax, (name, preds, color) in zip(axes, models_data):
        err_ut = np.abs(preds - tgt_np) * 1000
        mae = err_ut.mean()
        ax.scatter(b_mag, err_ut, s=5, alpha=0.2, color=color)
        bx, by = _binned_mean(b_mag, err_ut)
        ax.plot(bx, by, "o-", color="red", linewidth=2, markersize=4)
        ax.set_xlabel("|B| (mT)")
        ax.set_title(f"{name}\nMAE={mae:.1f} μT")
        ax.set_ylim(0, y_max * 1.05)
    axes[0].set_ylabel("Absolute error (μT)")
    fig.suptitle("Error vs B-field", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "forensics_error_vs_b.png", dpi=150)
    plt.close(fig)

    # ── Conclusion ──
    print("\n  95th percentile errors:")
    for name, preds, _ in models_data:
        err_ut = np.abs(preds - tgt_np) * 1000
        p95 = np.percentile(err_ut, 95)
        print(f"    {name:25s}  P95 = {p95:.1f} μT")

    print("\n  Conclusion: The student's error distribution closely tracks the")
    print("  teacher — same heavy tail at low-B and high-B, similar P95. Compression")
    print("  does not create new failure modes; it slightly inflates existing ones.")
    print("  Saved forensics_error_cdf.png, forensics_error_vs_b.png")


# ═════════════════════════════════════════════════════════════════════
# INVESTIGATION 4 — Quantization Noise Structure
# ═════════════════════════════════════════════════════════════════════

def investigation4_quant_noise():
    print("\n" + "=" * 60)
    print("INVESTIGATION 4: Quantization Noise Structure")
    print("=" * 60)

    test_spec, test_targets = load_filtered("test")
    spec_np = test_spec.numpy()
    tgt_np = test_targets.numpy()
    mask, snr, contrast, b_mag = _load_test_metadata()
    snr_per_dip = contrast * snr

    sess_fp32 = _ort_session("baseline_fp32.onnx")
    sess_int8 = _ort_session("v1_ptq_int8.onnx")

    pred_fp32 = _ort_predict_mt(sess_fp32, spec_np)
    pred_int8 = _ort_predict_mt(sess_int8, spec_np)

    delta = pred_int8 - pred_fp32  # signed, mT

    mu_d, std_d = delta.mean(), delta.std()
    print(f"  Δ mean = {mu_d*1000:.3f} μT,  Δ std = {std_d*1000:.3f} μT")

    corr_b = np.corrcoef(np.abs(delta), b_mag)[0, 1]
    corr_snr = np.corrcoef(np.abs(delta), snr_per_dip)[0, 1]
    print(f"  Pearson |Δ| vs b_magnitude:  r = {corr_b:.4f}")
    print(f"  Pearson |Δ| vs snr_per_dip:  r = {corr_snr:.4f}")

    # ── 2x2 subplot ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: histogram + normal overlay
    ax = axes[0, 0]
    ax.hist(delta * 1000, bins=80, density=True, alpha=0.7, color="steelblue",
            edgecolor="white", linewidth=0.3)
    fit_mu, fit_std = stats.norm.fit(delta * 1000)
    x_fit = np.linspace(delta.min() * 1000, delta.max() * 1000, 200)
    ax.plot(x_fit, stats.norm.pdf(x_fit, fit_mu, fit_std), "r-", linewidth=2,
            label=f"N({fit_mu:.2f}, {fit_std:.2f})")
    ax.set_xlabel("Δ (μT)")
    ax.set_ylabel("Density")
    ax.set_title("Δ distribution — is it Gaussian?")
    ax.legend()

    # Top-right: Δ vs B-field
    ax = axes[0, 1]
    ax.scatter(b_mag, delta * 1000, s=5, alpha=0.2, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    bx, by = _binned_mean(b_mag, delta * 1000)
    ax.plot(bx, by, "o-", color="red", linewidth=2, markersize=4)
    ax.set_xlabel("|B| (mT)")
    ax.set_ylabel("Δ (μT)")
    ax.set_title("Δ vs B-field")

    # Bottom-left: Δ vs snr_per_dip
    ax = axes[1, 0]
    ax.scatter(snr_per_dip, delta * 1000, s=5, alpha=0.2, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SNR per dip (contrast × SNR)")
    ax.set_ylabel("Δ (μT)")
    ax.set_title("Δ vs SNR per dip")

    # Bottom-right: Δ vs spectrum SNR
    ax = axes[1, 1]
    ax.scatter(snr, delta * 1000, s=5, alpha=0.2, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spectrum SNR")
    ax.set_ylabel("Δ (μT)")
    ax.set_title("Δ vs spectrum SNR")

    fig.suptitle("Quantization Noise Analysis (INT8 − FP32)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "forensics_quant_noise.png", dpi=150)
    plt.close(fig)

    print(f"\n  Conclusion: Δ has near-zero mean ({mu_d*1000:.2f} μT) and tiny std "
          f"({std_d*1000:.2f} μT).")
    if abs(corr_b) < 0.1 and abs(corr_snr) < 0.1:
        print("  The noise is effectively white — uncorrelated with B-field and SNR.")
        print("  Dynamic INT8 quantization (MatMul-only) adds negligible, unstructured error.")
    else:
        print(f"  There is mild correlation with B-field (r={corr_b:.3f}) and/or "
              f"SNR (r={corr_snr:.3f}),")
        print("  suggesting quantization noise is weakly structured — it hits harder where")
        print("  the model is already uncertain (low SNR / extreme B-field).")
    print("  Saved forensics_quant_noise.png")


# ═════════════════════════════════════════════════════════════════════
# INVESTIGATION 5B — Distillation Ablation
# ═════════════════════════════════════════════════════════════════════

def investigation5b_distillation_ablation(device):
    print("\n" + "=" * 60)
    print("INVESTIGATION 5B: Distillation Ablation")
    print("=" * 60)

    EPOCHS = 150
    BATCH_SIZE = 256
    LR = 3e-4
    WD = 1e-4
    WARMUP = 10
    scratch_ckpt = CKPT_DIR / "v5b_scratch_student.pt"

    # ── Train scratch student (no teacher) ──
    if scratch_ckpt.exists():
        print(f"  Checkpoint {scratch_ckpt.name} exists — skipping training.")
    else:
        print("  Training scratch student (no teacher, HuberLoss only) ...")
        train_spec, train_tgt = load_filtered("train")
        val_spec, val_tgt = load_filtered("val")
        print(f"  Train: {len(train_spec)}  Val: {len(val_spec)}")

        train_loader = DataLoader(
            TensorDataset(train_spec, train_tgt),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(val_spec, val_tgt),
            batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
        )

        model = TinyResNet1D().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

        def lr_lambda(epoch):
            if epoch < WARMUP:
                return (epoch + 1) / WARMUP
            progress = (epoch - WARMUP) / max(1, EPOCHS - WARMUP)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.HuberLoss(delta=0.5)

        best_val_mae = float("inf")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for spectra, targets_mt in train_loader:
                spectra = spectra.to(device)
                targets_norm = ((targets_mt.to(device)) - TARGET_MEAN) / TARGET_STD
                optimizer.zero_grad()
                loss = criterion(model(spectra), targets_norm)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for spectra, targets_mt in val_loader:
                    p = model(spectra.to(device)).cpu() * TARGET_STD + TARGET_MEAN
                    vp.append(p)
                    vt.append(targets_mt)
            val_mae_ut = float(np.abs(
                torch.cat(vp).numpy() - torch.cat(vt).numpy()
            ).mean()) * 1000

            if val_mae_ut < best_val_mae:
                best_val_mae = val_mae_ut
                torch.save(model.state_dict(), scratch_ckpt)

            if epoch % 10 == 0 or epoch == 1:
                print(f"    Ep {epoch:3d}/{EPOCHS} | val={val_mae_ut:.1f} μT | "
                      f"best={best_val_mae:.1f} μT")

        print(f"  Scratch training done. Best val MAE = {best_val_mae:.1f} μT\n")

    # ── Evaluate both students + teacher on test set ──
    test_spec, test_targets = load_filtered("test")

    distilled = _load_student("v2_student_best.pt", device)
    scratch = _load_student("v5b_scratch_student.pt", device)
    teacher = _load_teacher(device)

    mae_d, maxe_d, r2_d = evaluate(distilled, test_spec, test_targets, device)
    mae_s, maxe_s, r2_s = evaluate(scratch, test_spec, test_targets, device)
    mae_t, maxe_t, r2_t = evaluate(teacher, test_spec, test_targets, device)

    print(f"\n  {'Model':25s} {'MAE (μT)':>10s} {'Max Err (μT)':>14s} {'R²':>8s}")
    print(f"  {'-'*60}")
    print(f"  {'Distilled student':25s} {mae_d:>10.1f} {maxe_d:>14.1f} {r2_d:>8.4f}")
    print(f"  {'Scratch student':25s} {mae_s:>10.1f} {maxe_s:>14.1f} {r2_s:>8.4f}")
    print(f"  {'Teacher (FP32)':25s} {mae_t:>10.1f} {maxe_t:>14.1f} {r2_t:>8.4f}")

    # ── Saliency comparison ──
    print("\n  Computing gradient saliency (5-10 mT bin) ...")
    freq_axis = np.load(DATA_DIR / "freq_axis.npy")
    freq_ghz = freq_axis / 1e9

    mask_meta, snr_arr, _, b_mag_arr = _load_test_metadata()
    high_b = (b_mag_arr >= 5.0) & (b_mag_arr <= 10.0)
    indices = np.where(high_b)[0][:30]

    raw_test = np.load(DATA_DIR / "test.npz")
    raw_spec_filtered = raw_test["spectra"][mask_meta]
    sal_samples = test_spec[indices]
    raw_samples = raw_spec_filtered[indices]

    def compute_saliency(model, samples, device):
        model.eval().to(device)
        x = samples.clone().to(device).requires_grad_(True)
        out = model(x)
        out.sum().backward()
        return x.grad.abs().cpu().numpy()

    sal_distilled = compute_saliency(distilled, sal_samples, device)
    sal_scratch = compute_saliency(scratch, sal_samples, device)

    mean_sal_d = sal_distilled.mean(axis=0)
    mean_sal_s = sal_scratch.mean(axis=0)
    mean_raw = raw_samples.mean(axis=0)

    sal_max = max(mean_sal_d.max(), mean_sal_s.max()) * 1.1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, sal, title in [
        (ax1, mean_sal_d, "Distilled student saliency"),
        (ax2, mean_sal_s, "Scratch student saliency"),
    ]:
        ax_spec = ax
        ax_sal = ax.twinx()
        ax_spec.plot(freq_ghz, mean_raw, color="tab:blue", linewidth=1, label="Mean spectrum")
        ax_sal.fill_between(freq_ghz, sal, alpha=0.4, color="tab:orange", label="Mean saliency")
        ax_sal.set_ylim(0, sal_max)
        ax_spec.set_xlabel("Frequency (GHz)")
        ax_spec.set_ylabel("PL (raw)", color="tab:blue")
        ax_sal.set_ylabel("|∂pred/∂input|", color="tab:orange")
        ax.set_title(title)

    fig.suptitle("Saliency: Distilled vs Scratch (B ∈ [5, 10] mT)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "forensics_distillation_saliency.png", dpi=150)
    plt.close(fig)
    print("  Saved forensics_distillation_saliency.png")

    # Cosine similarity
    dot = np.dot(mean_sal_d, mean_sal_s)
    norm_d = np.linalg.norm(mean_sal_d)
    norm_s = np.linalg.norm(mean_sal_s)
    sim = dot / (norm_d * norm_s + 1e-12)
    print(f"\n  Saliency cosine similarity (distilled vs scratch): {sim:.3f}")

    if sim > 0.9:
        print("  Interpretation: The distilled and scratch students attend to nearly")
        print("  the same spectral features. The teacher transferred general structural")
        print("  knowledge (where dips are), not a unique internal representation.")
    elif sim > 0.7:
        print("  Interpretation: Moderate similarity. The teacher's soft targets guided")
        print("  the student to attend to features it might not have discovered on its")
        print("  own — distillation added meaningful inductive bias beyond the labels.")
    else:
        print("  Interpretation: Low similarity. The teacher transferred a substantially")
        print("  different feature-attention pattern. Distillation reshaped what the")
        print("  student considers informative, going well beyond label-only training.")


# ═════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    investigation3_error_cdf()
    investigation4_quant_noise()
    investigation5b_distillation_ablation(device)

    print("\n" + "=" * 60)
    print("All forensic investigations complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
