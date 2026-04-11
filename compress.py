"""Part 2: Model Compression — main script."""

import torch

from dataset import load_filtered
from models import ResNet1D, TinyResNet1D
from utils import BASELINE, evaluate, measure_latency, score_s, print_benchmark_table


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device: {device}")

    test_spec, test_targets = load_filtered("test")
    print(f"Test samples loaded: {len(test_spec)}")
    print(f"\nBaseline metrics:")
    for k, v in BASELINE.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
