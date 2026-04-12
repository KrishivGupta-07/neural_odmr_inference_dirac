"""
Part 1B: Model Training & Evaluation
=====================================
Train CNN1D and DilatedResNet1D on ODMR spectra -> |B| (mT) regression.
Produces checkpoints, a results table, and Investigation-2 error plots.

Run on Kaggle:  %run train_models.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
CKPT_DIR  = REPO_ROOT / "checkpoints"
PLOT_DIR  = Path(__file__).parent / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 777]
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 0.5
NUM_WORKERS = 2

ARCHITECTURES = {
    "cnn1d": lambda: CNN1D(),
    "dilresnet": lambda: DilatedResNet1D(),
}


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════

class ODMRDataset(Dataset):
    def __init__(self, spectra: np.ndarray, b_magnitude: np.ndarray, metadata=None):
        self.spectra = torch.from_numpy(spectra).float().unsqueeze(1)   # (N,1,512)
        self.targets = torch.from_numpy(b_magnitude).float()            # (N,)
        self.metadata = metadata

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.spectra[idx], self.targets[idx]


def load_data():
    """Load splits, z-score normalise spectra with training stats, save norm params."""
    train_npz = np.load(DATA_DIR / "train.npz")
    val_npz   = np.load(DATA_DIR / "val.npz")
    test_npz  = np.load(DATA_DIR / "test.npz")

    spec_mean = float(train_npz["spectra"].mean())
    spec_std  = float(train_npz["spectra"].std())

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_DIR / "norm_params.json", "w") as f:
        json.dump({"mean": spec_mean, "std": spec_std}, f, indent=2)
    print(f"Normalization  mean={spec_mean:.6f}  std={spec_std:.6f}")
    print(f"Saved {CKPT_DIR / 'norm_params.json'}\n")

    def norm(s):
        return (s - spec_mean) / spec_std

    train_ds = ODMRDataset(norm(train_npz["spectra"]), train_npz["b_magnitude"])
    val_ds   = ODMRDataset(norm(val_npz["spectra"]),   val_npz["b_magnitude"])
    test_ds  = ODMRDataset(
        norm(test_npz["spectra"]),
        test_npz["b_magnitude"],
        metadata={k: test_npz[k] for k in ("b_magnitude", "snr", "linewidth", "contrast")},
    )
    return train_ds, val_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1 — CNN1D
# ═══════════════════════════════════════════════════════════════════════════

class CNN1D(nn.Module):
    """4 x (Conv1d stride-2 -> BN -> ReLU) + GAP + linear head."""

    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        layers = []
        in_ch = 1
        kernels = [7, 5, 3, 3]
        paddings = [3, 2, 1, 1]
        for out_ch, k, p in zip(channels, kernels, paddings):
            layers += [
                nn.Conv1d(in_ch, out_ch, k, stride=2, padding=p),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.pool(self.features(x)).squeeze(-1)
        return self.head(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2 — DilatedResNet1D
# ═══════════════════════════════════════════════════════════════════════════

class _DilResBlock(nn.Module):
    """Pre-activation residual block with dilated convolutions."""

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
    """Channel expansion + stride-2 downsampling."""

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
    """Dilated-conv residual net: wide receptive field captures full dip span.

    6 dilated-res blocks (dilation 1,2,4,8,16,32) give an RF contribution
    of ~252 bins before downsampling.  Two stride-2 down-blocks and global
    average pooling let the head aggregate the full 512-point spectrum.
    """

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
            _DownBlock(base_ch, base_ch * 2),      # 512 -> 256
            _DownBlock(base_ch * 2, base_ch * 4),   # 256 -> 128
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
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(arch_name, seed, train_ds, val_ds):
    """Train one model, return path to best checkpoint and training history."""
    seed_everything(seed)
    model = ARCHITECTURES[arch_name]().to(DEVICE)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    best_val_mae = float("inf")
    ckpt_path = CKPT_DIR / f"{arch_name}_seed{seed}.pt"
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        running_loss = 0.0
        for spectra, targets in train_loader:
            spectra, targets = spectra.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(spectra)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)
        train_loss = running_loss / len(train_ds)

        # ── validate ──
        model.eval()
        val_preds, val_targets = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for spectra, targets in val_loader:
                spectra, targets = spectra.to(DEVICE), targets.to(DEVICE)
                preds = model(spectra)
                val_running_loss += criterion(preds, targets).item() * len(targets)
                val_preds.append(preds.cpu())
                val_targets.append(targets.cpu())
        val_loss = val_running_loss / len(val_ds)
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_mae = float(np.mean(np.abs(val_preds - val_targets)))

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                f"val_MAE={val_mae * 1000:.1f} μT  lr={lr_now:.2e}"
                f"{'  *best*' if val_mae <= best_val_mae else ''}"
            )

    print(f"  Best val MAE = {best_val_mae * 1000:.1f} μT  ->  {ckpt_path.name}\n")
    return ckpt_path, best_val_mae, history


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency(model, input_shape=(1, 1, 512), n_runs=200, warmup=20):
    """Mean CPU inference latency in ms for a single sample."""
    model_cpu = model.cpu().eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        for _ in range(warmup):
            model_cpu(x)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model_cpu(x)
        elapsed = (time.perf_counter() - t0) / n_runs * 1000
    return elapsed


def evaluate_checkpoint(arch_name, ckpt_path, test_loader, test_ds):
    """Load checkpoint, evaluate on test set, return metrics dict."""
    model = ARCHITECTURES[arch_name]().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for spectra, targets in test_loader:
            spectra = spectra.to(DEVICE)
            preds = model(spectra)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    preds_mT = np.concatenate(all_preds)
    targets_mT = np.concatenate(all_targets)

    preds_uT = preds_mT * 1000
    targets_uT = targets_mT * 1000

    mae_uT = float(np.mean(np.abs(preds_uT - targets_uT)))
    max_err_uT = float(np.max(np.abs(preds_uT - targets_uT)))
    r2 = float(r2_score(targets_mT, preds_mT))
    n_params = count_params(model)
    file_kb = os.path.getsize(ckpt_path) / 1024
    latency_ms = measure_latency(model)
    model.to(DEVICE)

    return {
        "arch": arch_name,
        "ckpt": ckpt_path.name,
        "mae_uT": mae_uT,
        "max_err_uT": max_err_uT,
        "r2": r2,
        "params": n_params,
        "size_kb": file_kb,
        "latency_ms": latency_ms,
        "preds_mT": preds_mT,
        "targets_mT": targets_mT,
    }


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGATION 2 PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_investigation2(preds_mT, targets_mT, test_ds, tag="best"):
    """Abs-error vs B-field and vs SNR, with binned-mean overlay."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    abs_err = np.abs(preds_mT - targets_mT)
    meta = test_ds.metadata
    b_mag = meta["b_magnitude"]
    snr = meta["snr"]

    for x_arr, x_label, x_tag, n_bins in [
        (b_mag, "|B| (mT)", "bfield", 25),
        (snr,   "SNR",      "snr",    25),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(x_arr, abs_err, s=2, alpha=0.25, color="steelblue", label="Per-sample")

        bin_edges = np.linspace(x_arr.min(), x_arr.max(), n_bins + 1)
        bin_centers, bin_means = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (x_arr >= lo) & (x_arr < hi)
            if mask.sum() > 0:
                bin_centers.append((lo + hi) / 2)
                bin_means.append(abs_err[mask].mean())
        ax.plot(bin_centers, bin_means, "o-", color="#e74c3c", linewidth=2,
                markersize=5, label="Binned mean")

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Absolute Error (mT)", fontsize=11)
        ax.set_title(f"Test Error vs {x_label}  [{tag}]", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = PLOT_DIR / f"error_vs_{x_tag}_{tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    # ── load data ──
    train_ds, val_ds, test_ds = load_data()
    print(f"Train {len(train_ds)}  Val {len(val_ds)}  Test {len(test_ds)}\n")

    # ── train all runs ──
    run_info = []
    for arch_name in ARCHITECTURES:
        for seed in SEEDS:
            print(f"{'='*60}")
            print(f"Training {arch_name}  seed={seed}")
            print(f"{'='*60}")
            ckpt_path, best_mae, hist = train_model(arch_name, seed, train_ds, val_ds)
            run_info.append((arch_name, seed, ckpt_path))

    # ── evaluate all checkpoints on test set ──
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    results = []
    for arch_name, seed, ckpt_path in run_info:
        metrics = evaluate_checkpoint(arch_name, ckpt_path, test_loader, test_ds)
        metrics["seed"] = seed
        results.append(metrics)

    # ── print results table ──
    print("\n" + "=" * 100)
    print("TEST-SET RESULTS")
    print("=" * 100)
    header = (
        f"{'Model':<14} {'Seed':>5} {'MAE(μT)':>10} {'MaxErr(μT)':>12} "
        f"{'R²':>8} {'Params':>10} {'Size(KB)':>10} {'Latency(ms)':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['arch']:<14} {r['seed']:>5} {r['mae_uT']:>10.1f} {r['max_err_uT']:>12.1f} "
            f"{r['r2']:>8.5f} {r['params']:>10,} {r['size_kb']:>10.1f} {r['latency_ms']:>12.2f}"
        )
    print("-" * len(header))

    # ── per-architecture summary ──
    for arch in ARCHITECTURES:
        arch_results = [r for r in results if r["arch"] == arch]
        maes = [r["mae_uT"] for r in arch_results]
        print(f"  {arch:14s}  MAE mean={np.mean(maes):.1f} μT  std={np.std(maes):.1f} μT")
    print()

    # ── Investigation 2: error plots for the single best model ──
    best = min(results, key=lambda r: r["mae_uT"])
    print(f"Best model: {best['ckpt']}  MAE={best['mae_uT']:.1f} μT\n")
    print("Investigation 2 — error scatter plots:")
    plot_investigation2(best["preds_mT"], best["targets_mT"], test_ds, tag="best")

    print("\nDone.")


if __name__ == "__main__":
    main()
