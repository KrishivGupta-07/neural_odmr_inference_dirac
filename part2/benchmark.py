"""Part 2 — Final benchmark report for submission."""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

ROOT = Path(__file__).parent
BENCHMARK_CSV = ROOT / "benchmark_table.csv"

EXPECTED_FILES = [
    "part2/benchmark_table.csv",
    "part2/models/baseline_fp32.onnx",
    "part2/models/v1_ptq_int8.onnx",
    "part2/models/v2_student_fp32.onnx",
    "part2/models/v3_student_int8.onnx",
    "part2/models/v4_pruned70.onnx",
    "checkpoints/v2_student_best.pt",
    "checkpoints/v4_pruned70_best.pt",
    "checkpoints/v5b_scratch_student.pt",
    "part2/figures/pareto_mae_vs_size.png",
    "part2/figures/pareto_mae_vs_latency.png",
    "part2/figures/compression_scores.png",
    "part2/figures/forensics_error_cdf.png",
    "part2/figures/forensics_error_vs_b.png",
    "part2/figures/forensics_quant_noise.png",
    "part2/figures/forensics_distillation_saliency.png",
]


def load_rows():
    with open(BENCHMARK_CSV, newline="") as f:
        return list(csv.DictReader(f))


def print_benchmark_table(rows, out):
    fmt = (
        "  {tag:<26s} {mae:>10s} {maxe:>13s} {r2:>8s} "
        "{params:>8s} {size:>10s} {lat:>12s} {score:>8s}"
    )
    header = fmt.format(
        tag="Variant", mae="MAE(μT)", maxe="MaxErr(μT)", r2="R²",
        params="Params", size="Size(KB)", lat="Latency(ms)", score="Score S",
    )
    out.write(header + "\n")
    out.write("  " + "-" * (len(header) - 2) + "\n")
    for r in rows:
        out.write(fmt.format(
            tag=r["tag"],
            mae=r["mae_ut"],
            maxe=r["max_err_ut"],
            r2=r["r2"],
            params=r["params"],
            size=r["size_kb"],
            lat=r["latency_ms"],
            score=r["score"],
        ) + "\n")


def write_report(rows, out):
    bl = next(r for r in rows if "Baseline" in r["tag"])
    v1 = next(r for r in rows if "V1" in r["tag"])
    v2 = next(r for r in rows if r["tag"].startswith("V2"))
    v3 = next(r for r in rows if "V3" in r["tag"])
    v4 = next(r for r in rows if "V4" in r["tag"])

    bl_mae = float(bl["mae_ut"])
    bl_size = float(bl["size_kb"])
    bl_lat = float(bl["latency_ms"])

    v2_mae = float(v2["mae_ut"])
    v2_size = float(v2["size_kb"])
    v2_lat = float(v2["latency_ms"])
    v2_score = float(v2["score"])

    size_ratio = bl_size / v2_size
    lat_ratio = bl_lat / v2_lat
    mae_pct = (v2_mae - bl_mae) / bl_mae * 100

    out.write("=" * 70 + "\n")
    out.write("  PART 2 — COMPRESSION SCIENCE: FULL BENCHMARK REPORT\n")
    out.write("=" * 70 + "\n\n")

    out.write("BENCHMARK TABLE\n")
    out.write("=" * 70 + "\n")
    print_benchmark_table(rows, out)

    out.write(f"\n\nCOMPRESSION FINDINGS\n")
    out.write("=" * 70 + "\n")
    out.write(f"  Best compression score:    V2 Distilled Student  S={v2_score:.0f}\n")
    out.write(f"  Best size reduction:       V2  {bl_size}→{v2_size} KB  ({size_ratio:.1f}x)\n")
    out.write(f"  Best latency reduction:    V2  {bl_lat}→{v2_lat} ms ({lat_ratio:.1f}x)\n")
    out.write(f"  MAE degradation (best):    V2  {bl_mae}→{v2_mae} μT (+{mae_pct:.1f}%)\n")
    out.write(f"\n")
    out.write(f"  Pareto-optimal variants:   Baseline, V1, V2, V4\n")
    out.write(f"  Recommended deployment:    V2 Distilled Student\n")

    out.write(f"\n\nFORENSICS FINDINGS\n")
    out.write("=" * 70 + "\n")
    out.write(
        "  Inv 3 — Error distribution: compression preserves error shape,\n"
        "           no new failure modes. P95: Baseline=1847μT Student=1887μT\n"
    )
    out.write(
        "  Inv 4 — Quant noise: Gaussian white noise (mean=1.74μT std=5.30μT),\n"
        "           zero correlation with B-field or SNR. INT8 is safe.\n"
    )
    out.write(
        "  Inv 5B — Distillation: saliency similarity=0.967, teacher provides\n"
        "            label smoothing not feature transfer. Δ MAE = 40μT (6.3%)\n"
    )

    out.write(f"\n\nCOMPRESSION SCORE EXPLANATION\n")
    out.write("=" * 70 + "\n")

    out.write(f"\n  Why V2 >> V3 (Student+INT8):\n")
    out.write(
        "    Quantizing the tiny student adds metadata overhead (+3 KB) and\n"
        "    doesn't reduce latency further because the linear layers are already\n"
        "    too small to benefit from INT8 MatMul. Architecture reduction\n"
        "    dominates quantization at this scale.\n"
    )

    out.write(f"\n  Why V4 (Pruning) scores only {float(v4['score']):.1f}:\n")
    out.write(
        "    Unstructured magnitude pruning creates zero weights but they remain\n"
        "    stored in the ONNX graph. No size reduction, modest latency gain\n"
        "    from ORT skipping near-zero multiplications. Structured pruning\n"
        "    or sparse tensor formats would be required to realize size savings.\n"
    )

    out.write(f"\n  Why V1 (PTQ INT8) scores only {float(v1['score']):.1f}:\n")
    out.write(
        "    Dynamic quantization only affects MatMul/Gemm ops (the linear head).\n"
        "    Conv weights — 95% of parameters — remain FP32 in ONNX dynamic quant.\n"
        "    Score comes from ORT graph optimization, not quantization itself.\n"
    )


def check_files(out):
    out.write(f"\n\nOUTPUT FILE CHECKLIST\n")
    out.write("=" * 70 + "\n")
    all_ok = True
    for rel in EXPECTED_FILES:
        exists = (REPO_ROOT / rel).exists()
        mark = "✓" if exists else "✗"
        out.write(f"  [{mark}] {rel}\n")
        if not exists:
            all_ok = False
    if all_ok:
        out.write("\n  All 16 files present.\n")
    else:
        out.write("\n  WARNING: some files missing.\n")
    return all_ok


def main():
    rows = load_rows()
    report_path = REPO_ROOT / "compression_report.txt"

    with open(report_path, "w") as f:
        class Tee:
            def write(self, s):
                sys.stdout.write(s)
                f.write(s)
            def flush(self):
                sys.stdout.flush()
                f.flush()

        out = Tee()
        write_report(rows, out)
        check_files(out)

    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
