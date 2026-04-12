"""Part 1B — Train ResNet1D(base_ch=32) across 3 seeds, evaluate, plot curves."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
PLOT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

TARGET_MEAN = 4.8513
TARGET_STD  = 2.3309
SIGMA       = 4
SNR_THRESH  = 5.0
SEEDS       = [42, 123, 777]

EPOCHS     = 150
BATCH_SIZE = 256
LR         = 3e-4
WD         = 1e-4
WARMUP     = 10
GRAD_CLIP  = 1.0


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(split: str):
    """Load a data split, filter by SNR threshold and anomaly flag, preprocess."""
    npz = np.load(DATA_DIR / f"{split}.npz")
    spectra  = npz["spectra"]
    b_mag    = npz["b_magnitude"]
    contrast = npz["contrast"]
    snr      = npz["snr"]
    anomaly  = npz["is_anomalous"].astype(bool)

    mask = (contrast * snr >= SNR_THRESH) & (~anomaly)
    spectra = preprocess_spectra(spectra[mask], sigma=SIGMA)
    b_mag   = b_mag[mask].astype(np.float32)

    return torch.from_numpy(spectra), torch.from_numpy(b_mag)


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def val_mae_ut(model, loader, device):
    """Compute validation MAE in micro-Tesla."""
    model.eval()
    preds, tgts = [], []
    for spec, tgt_mt in loader:
        out_z = model(spec.to(device)).cpu()
        preds.append(out_z * TARGET_STD + TARGET_MEAN)
        tgts.append(tgt_mt)
    preds_mt = torch.cat(preds).numpy()
    tgts_mt  = torch.cat(tgts).numpy()
    return float(np.abs(preds_mt - tgts_mt).mean()) * 1000


def train_one_seed(seed, train_spec, train_tgt, val_spec, val_tgt, device):
    """Train ResNet1D for one seed. Returns (ckpt_path, val_history)."""
    seed_everything(seed)
    ckpt_path = CKPT_DIR / f"resnet_smooth_seed{seed}_best.pt"

    train_norm = (train_tgt - TARGET_MEAN) / TARGET_STD
    train_loader = DataLoader(
        TensorDataset(train_spec, train_tgt, train_norm),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_spec, val_tgt),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
    )

    model = ResNet1D(base_ch=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    def lr_lambda(epoch):
        if epoch < WARMUP:
            return (epoch + 1) / WARMUP
        progress = (epoch - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.HuberLoss(delta=0.5)

    best_val = float("inf")
    val_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for spec, _tgt_mt, tgt_norm in train_loader:
            spec = spec.to(device)
            tgt_norm = tgt_norm.to(device)

            optimizer.zero_grad()
            loss = criterion(model(spec), tgt_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item() * len(tgt_norm)
            n += len(tgt_norm)

        scheduler.step()
        avg_loss = running_loss / n

        v = val_mae_ut(model, val_loader, device)
        val_history.append(v)

        if v < best_val:
            best_val = v
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{EPOCHS} | loss={avg_loss:.5f} | "
                  f"val={v:.1f} μT | best={best_val:.1f} μT")

    print(f"  Best val MAE = {best_val:.1f} μT  →  {ckpt_path.name}\n")
    return ckpt_path, val_history, best_val


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, test_spec, test_tgt, device):
    """Evaluate a checkpoint on test set. Returns (mae_ut, max_err_ut, r2)."""
    model = ResNet1D(base_ch=32).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(
        TensorDataset(test_spec, test_tgt),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
    )
    preds, tgts = [], []
    for spec, tgt_mt in loader:
        out_z = model(spec.to(device)).cpu()
        preds.append(out_z * TARGET_STD + TARGET_MEAN)
        tgts.append(tgt_mt)

    preds_mt = torch.cat(preds).numpy()
    tgts_mt  = torch.cat(tgts).numpy()

    abs_err = np.abs(preds_mt - tgts_mt)
    mae_ut     = float(abs_err.mean()) * 1000
    max_err_ut = float(abs_err.max()) * 1000
    ss_res = np.sum((tgts_mt - preds_mt) ** 2)
    ss_tot = np.sum((tgts_mt - tgts_mt.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)

    return mae_ut, max_err_ut, r2


def plot_training_curves(all_histories, all_bests):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, seed, hist, best in zip(axes, SEEDS, all_histories, all_bests):
        ax.plot(range(1, EPOCHS + 1), hist, linewidth=1.2, color="tab:blue")
        ax.axhline(best, color="tab:red", linestyle="--", linewidth=0.8,
                    label=f"best = {best:.1f} μT")
        ax.set_xlabel("Epoch")
        ax.set_title(f"Seed {seed} — best val MAE = {best:.1f} μT")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Val MAE (μT)")
    fig.suptitle("Training Curves — ResNet1D(base_ch=32)", fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / "training_curves_all_seeds.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    device = get_device()
    print(f"Device: {device}")

    print("Loading data ...")
    train_spec, train_tgt = load_split("train")
    val_spec, val_tgt     = load_split("val")
    test_spec, test_tgt   = load_split("test")
    print(f"  Train: {len(train_spec)}  Val: {len(val_spec)}  Test: {len(test_spec)}\n")

    all_histories = []
    all_bests = []
    ckpt_paths = []

    for seed in SEEDS:
        print(f"{'='*60}")
        print(f"Training ResNet1D  seed={seed}")
        print(f"{'='*60}")
        ckpt, hist, best = train_one_seed(
            seed, train_spec, train_tgt, val_spec, val_tgt, device)
        ckpt_paths.append(ckpt)
        all_histories.append(hist)
        all_bests.append(best)

    # ── Evaluate all checkpoints ──
    results = []
    for seed, ckpt in zip(SEEDS, ckpt_paths):
        mae, maxe, r2 = evaluate_checkpoint(ckpt, test_spec, test_tgt, device)
        results.append((seed, mae, maxe, r2))

    maes  = [r[1] for r in results]
    maxes = [r[2] for r in results]
    r2s   = [r[3] for r in results]

    # ── Print final table ──
    print()
    print("╔══════════╦══════════════╦═══════════════╦═══════════════╗")
    print("║           FINAL RESULTS — ALL 3 SEEDS                  ║")
    print("╠══════════╬══════════════╬═══════════════╬═══════════════╣")
    print("║ Seed     ║ Test MAE μT  ║ Max Err μT    ║ R²            ║")
    print("╠══════════╬══════════════╬═══════════════╬═══════════════╣")
    for seed, mae, maxe, r2 in results:
        print(f"║ {seed:<8d} ║ {mae:>10.1f}   ║ {maxe:>11.1f}   ║ {r2:>10.4f}    ║")
    print("╠══════════╬══════════════╬═══════════════╬═══════════════╣")
    m_mae, s_mae = np.mean(maes), np.std(maes)
    m_max, s_max = np.mean(maxes), np.std(maxes)
    m_r2, s_r2   = np.mean(r2s), np.std(r2s)
    print(f"║ Mean±Std ║ {m_mae:.1f}±{s_mae:.1f}   ║ {m_max:.1f}±{s_max:.1f}  ║ {m_r2:.4f}±{s_r2:.4f}║")
    print("╚══════════╩══════════════╩═══════════════╩═══════════════╝")

    # ── Plot training curves ──
    plot_training_curves(all_histories, all_bests)

    # ── Save summary JSON ──
    summary = {
        "seeds": SEEDS,
        "mae_ut_mean": round(m_mae, 2),
        "mae_ut_std": round(s_mae, 2),
        "max_err_ut_mean": round(m_max, 2),
        "max_err_ut_std": round(s_max, 2),
        "r2_mean": round(m_r2, 5),
        "r2_std": round(s_r2, 5),
        "per_seed": [
            {"seed": s, "mae_ut": round(m, 2), "max_err_ut": round(mx, 2), "r2": round(r, 5)}
            for s, m, mx, r in results
        ],
    }
    json_path = CKPT_DIR / "training_summary_all_seeds.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
