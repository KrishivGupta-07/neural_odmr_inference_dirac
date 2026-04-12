"""Architecture definitions for ODMR B-field regression."""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Post-activation residual: relu(x + net(x))."""

    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class _DownBlock(nn.Module):
    """Stride-2 downsampling with learned skip projection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        return torch.relu(self.conv(x) + self.skip(x))


class ResNet1D(nn.Module):
    """Baseline 1D ResNet for ODMR regression. 186 529 params at base_ch=32."""

    def __init__(self, base_ch=32):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, 7, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(_ResBlock(c), _ResBlock(c))
        self.stage2 = nn.Sequential(_DownBlock(c, c * 2), _ResBlock(c * 2))
        self.stage3 = nn.Sequential(_DownBlock(c * 2, c * 4), _ResBlock(c * 4))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(c * 4, c * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c * 2, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class TinyResNet1D(ResNet1D):
    """Student model — identical topology, base_ch=8."""

    def __init__(self):
        super().__init__(base_ch=8)
