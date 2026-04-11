"""
Overfit sanity check — run this BEFORE full training.
Trains DilatedResNet1D on 32 samples for 500 epochs to confirm the model
can memorise a single batch. No saved weights needed.

Run on Kaggle:  %run overfit_test.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── paths (same as train_models.py) ──
DATA_DIR = Path("/kaggle/input/datasets/krishivgupta123123/diraclab/data")
CKPT_DIR = Path("/kaggle/working/checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── model definition (copied from train_models.py for self-containment) ──

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


# ── run test ──

def main():
    print(f"Device: {DEVICE}")

    train_npz = np.load(DATA_DIR / "train.npz")
    spec_mean = float(train_npz["spectra"].mean())
    spec_std = float(train_npz["spectra"].std())
    print(f"Normalization  mean={spec_mean:.6f}  std={spec_std:.6f}")

    spectra = (train_npz["spectra"][:32] - spec_mean) / spec_std
    targets = train_npz["b_magnitude"][:32]

    spectra_batch = torch.from_numpy(spectra).float().unsqueeze(1).to(DEVICE)
    targets_batch = torch.from_numpy(targets).float().to(DEVICE)

    model = DilatedResNet1D().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"\n{'='*60}")
    print(f"OVERFIT TEST  (32 samples, 500 epochs, MSELoss)")
    print(f"{'='*60}")

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        preds = model(spectra_batch)
        loss = loss_fn(preds, targets_batch)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            mae_ut = (preds - targets_batch).abs().mean().item() * 1000
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f} | MAE: {mae_ut:.1f} μT")

    model.eval()
    with torch.no_grad():
        preds = model(spectra_batch)
    final_mae = (preds - targets_batch).abs().mean().item() * 1000

    print(f"\nSample predictions vs targets:")
    for i in range(5):
        print(f"  pred={preds[i].item():.4f} mT | true={targets_batch[i].item():.4f} mT")

    print(f"\nFinal MAE: {final_mae:.1f} μT")
    if final_mae > 50:
        print(f"WARNING: model could not memorise 32 samples — check architecture.")
    else:
        print(f"PASS: model memorised the batch.")


if __name__ == "__main__":
    main()
