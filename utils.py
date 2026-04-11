"""Evaluation, latency measurement, scoring, and CSV logging."""

import csv
import time
from pathlib import Path

import numpy as np
import torch

BASELINE = dict(mae_ut=561.3, size_kb=764.9, latency_ms=1.990, params=186529)
TARGET_MEAN = 4.8513
TARGET_STD = 2.3309

BENCHMARK_CSV = Path(__file__).parent / "benchmark.csv"
_CSV_FIELDS = [
    "tag", "mae_ut", "max_err_ut", "r2", "params", "size_kb",
    "latency_ms", "score", "notes",
]


@torch.no_grad()
def evaluate(model, spectra_t, targets_mt, device, batch_size=512):
    """Evaluate model on pre-processed spectra. Targets are in raw mT.

    The model outputs z-scored predictions; this function un-normalises them
    before computing metrics.

    Returns (mae_ut, max_err_ut, r2).
    """
    model.eval()
    preds = []
    for i in range(0, len(spectra_t), batch_size):
        batch = spectra_t[i : i + batch_size].to(device)
        out = model(batch).cpu()
        preds.append(out)
    preds_z = torch.cat(preds).numpy()
    preds_mt = preds_z * TARGET_STD + TARGET_MEAN
    targets = targets_mt.numpy()

    abs_err_mt = np.abs(preds_mt - targets)
    mae_ut = float(abs_err_mt.mean()) * 1000
    max_err_ut = float(abs_err_mt.max()) * 1000
    ss_res = np.sum((targets - preds_mt) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    return mae_ut, max_err_ut, r2


def measure_latency(model, n_runs=1000):
    """Mean single-sample CPU inference latency in milliseconds."""
    m = model.cpu().eval()
    x = torch.randn(1, 512)
    with torch.no_grad():
        for _ in range(100):
            m(x)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            m(x)
        elapsed = (time.perf_counter() - t0) / n_runs * 1000
    return elapsed


def score_s(mae_ut, size_kb, latency_ms):
    """Compression score: S = (baseline_mae/mae) * (baseline_size/size) * (baseline_lat/lat)."""
    b = BASELINE
    return (b["mae_ut"] / mae_ut) * (b["size_kb"] / size_kb) * (b["latency_ms"] / latency_ms)


def write_benchmark_row(row_dict):
    """Append one row to benchmark.csv, creating it with a header if needed."""
    exists = BENCHMARK_CSV.exists()
    with open(BENCHMARK_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


def print_benchmark_table():
    """Read benchmark.csv and print a formatted table."""
    if not BENCHMARK_CSV.exists():
        print("No benchmark.csv found.")
        return
    with open(BENCHMARK_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("benchmark.csv is empty.")
        return

    fmt = (
        "{tag:<30s} {mae_ut:>10s} {max_err_ut:>12s} {r2:>8s} "
        "{params:>8s} {size_kb:>10s} {latency_ms:>12s} {score:>8s}"
    )
    header = fmt.format(
        tag="Tag", mae_ut="MAE(μT)", max_err_ut="MaxErr(μT)", r2="R²",
        params="Params", size_kb="Size(KB)", latency_ms="Latency(ms)", score="Score",
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(fmt.format(
            tag=r.get("tag", ""),
            mae_ut=r.get("mae_ut", ""),
            max_err_ut=r.get("max_err_ut", ""),
            r2=r.get("r2", ""),
            params=r.get("params", ""),
            size_kb=r.get("size_kb", ""),
            latency_ms=r.get("latency_ms", ""),
            score=r.get("score", ""),
        ))
