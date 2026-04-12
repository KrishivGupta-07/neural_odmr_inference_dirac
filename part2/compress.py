"""Part 2: Model Compression — main script."""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

from dataset import load_filtered
from models import ResNet1D, TinyResNet1D
from utils import (
    BASELINE, BENCHMARK_CSV, TARGET_MEAN, TARGET_STD,
    evaluate, measure_latency, score_s,
    write_benchmark_row, print_benchmark_table,
)

PLOT_DIR = Path(__file__).parent / "figures"
CKPT_DIR = REPO_ROOT / "checkpoints"
ONNX_DIR = Path(__file__).parent / "models"
BASELINE_CKPT = CKPT_DIR / "resnet_smooth_seed42_best.pt"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_baseline_model():
    model = ResNet1D(base_ch=32)
    sd = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


def _onnx_evaluate(session, spectra_np, targets_mt_np, batch_size=512):
    """Run ORT inference, denormalize, compute metrics. Returns (mae_ut, max_err_ut, r2)."""
    input_name = session.get_inputs()[0].name
    preds = []
    for i in range(0, len(spectra_np), batch_size):
        batch = spectra_np[i : i + batch_size]
        out = session.run(None, {input_name: batch})[0].reshape(-1)
        preds.append(out)
    preds_z = np.concatenate(preds)
    preds_mt = preds_z * TARGET_STD + TARGET_MEAN

    abs_err = np.abs(preds_mt - targets_mt_np)
    mae_ut = float(abs_err.mean()) * 1000
    max_err_ut = float(abs_err.max()) * 1000
    ss_res = np.sum((targets_mt_np - preds_mt) ** 2)
    ss_tot = np.sum((targets_mt_np - targets_mt_np.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    return mae_ut, max_err_ut, r2


def _onnx_latency(session, n_runs=1000):
    """Single-sample CPU latency in ms."""
    input_name = session.get_inputs()[0].name
    x = np.random.randn(1, 512).astype(np.float32)
    for _ in range(100):
        session.run(None, {input_name: x})
    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {input_name: x})
    return (time.perf_counter() - t0) / n_runs * 1000


# ═════════════════════════════════════════════════════════════════════
# VARIANT 1 — Post-Training Quantization (INT8 dynamic, ONNX)
# ═════════════════════════════════════════════════════════════════════

def variant1_ptq_int8(test_spec, test_targets):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = ONNX_DIR / "baseline_fp32.onnx"
    int8_path = ONNX_DIR / "v1_ptq_int8.onnx"

    print("\n" + "=" * 60)
    print("VARIANT 1: PTQ INT8 (ONNX dynamic quantization)")
    print("=" * 60)

    # ── 1. Export FP32 ONNX ──
    model = _load_baseline_model()
    dummy = torch.randn(1, 512)
    torch.onnx.export(
        model, dummy, str(fp32_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    print(f"  Exported FP32 ONNX -> {fp32_path}  ({os.path.getsize(fp32_path)/1024:.1f} KB)")

    # ── 2. Verify ONNX vs PyTorch ──
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess_fp32 = ort.InferenceSession(str(fp32_path), opts, providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    check_np = test_spec[:10].numpy()
    with torch.no_grad():
        pt_out = model(test_spec[:10]).numpy()
    ort_out = sess_fp32.run(None, {input_name: check_np})[0].reshape(-1)
    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f"  FP32 ONNX vs PyTorch max diff: {max_diff:.6f}  {'OK' if max_diff < 0.01 else 'FAIL'}")

    # ── 3. Quantize to INT8 ──
    quantize_dynamic(
        str(fp32_path), str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    int8_kb = os.path.getsize(int8_path) / 1024
    print(f"  Quantized INT8 -> {int8_path}  ({int8_kb:.1f} KB)")

    # ── 4. Evaluate INT8 ──
    sess_int8 = ort.InferenceSession(str(int8_path), opts, providers=["CPUExecutionProvider"])
    spec_np = test_spec.numpy()
    tgt_np = test_targets.numpy()

    mae_ut, max_err_ut, r2 = _onnx_evaluate(sess_int8, spec_np, tgt_np)
    lat_ms = _onnx_latency(sess_int8)
    s = score_s(mae_ut, int8_kb, lat_ms)

    print(f"\n  V1 PTQ INT8 — MAE={mae_ut:.1f} μT | Size={int8_kb:.1f} KB | "
          f"Latency={lat_ms:.3f} ms | S={s:.2f}")

    # ── 5. Write to benchmark ──
    if not BENCHMARK_CSV.exists():
        write_benchmark_row(dict(
            tag="Baseline (FP32 PyTorch)",
            mae_ut=f"{BASELINE['mae_ut']:.1f}",
            max_err_ut="—",
            r2="0.8551",
            params=str(BASELINE["params"]),
            size_kb=f"{BASELINE['size_kb']:.1f}",
            latency_ms=f"{BASELINE['latency_ms']:.3f}",
            score="1.00",
            notes="ResNet1D base_ch=32",
        ))

    write_benchmark_row(dict(
        tag="V1 PTQ INT8",
        mae_ut=f"{mae_ut:.1f}",
        max_err_ut=f"{max_err_ut:.1f}",
        r2=f"{r2:.4f}",
        params=str(BASELINE["params"]),
        size_kb=f"{int8_kb:.1f}",
        latency_ms=f"{lat_ms:.3f}",
        score=f"{s:.2f}",
        notes="ONNX dynamic quant QInt8",
    ))

    print()
    print_benchmark_table()


# ═════════════════════════════════════════════════════════════════════
# VARIANT 2 — Knowledge Distillation (TinyResNet1D)
# ═════════════════════════════════════════════════════════════════════

def variant2_distillation(test_spec, test_targets, device):
    import math
    import onnxruntime as ort
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn as nn

    print("\n" + "=" * 60)
    print("VARIANT 2: Knowledge Distillation")
    print("=" * 60)

    EPOCHS = 150
    BATCH_SIZE = 256
    LR = 3e-4
    WD = 1e-4
    WARMUP = 10
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    student_ckpt = CKPT_DIR / "v2_student_best.pt"
    n_params = sum(p.numel() for p in TinyResNet1D().parameters())

    if student_ckpt.exists():
        print(f"  Checkpoint {student_ckpt.name} exists — skipping training.")
    else:
        train_spec, train_tgt = load_filtered("train")
        val_spec, val_tgt = load_filtered("val")
        print(f"  Train: {len(train_spec)}  Val: {len(val_spec)}  Test: {len(test_spec)}")

        train_ds = TensorDataset(train_spec, train_tgt)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=0)
        val_ds = TensorDataset(val_spec, val_tgt)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                                num_workers=0)

        teacher = _load_baseline_model().to(device)
        teacher.eval()

        student = TinyResNet1D().to(device)
        n_params = sum(p.numel() for p in student.parameters())
        print(f"  Student params: {n_params:,}")

        optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WD)

        def lr_lambda(epoch):
            if epoch < WARMUP:
                return (epoch + 1) / WARMUP
            progress = (epoch - WARMUP) / max(1, EPOCHS - WARMUP)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        loss_hard_fn = nn.HuberLoss(delta=0.5)
        loss_soft_fn = nn.MSELoss()

        best_val_mae = float("inf")
        for epoch in range(1, EPOCHS + 1):
            student.train()
            running_loss = 0.0
            n_samples = 0
            for spectra, targets_mt in train_loader:
                spectra = spectra.to(device)
                targets_mt = targets_mt.to(device)
                targets_norm = (targets_mt - TARGET_MEAN) / TARGET_STD

                with torch.no_grad():
                    teacher_preds = teacher(spectra)
                student_preds = student(spectra)

                loss_hard = loss_hard_fn(student_preds, targets_norm)
                loss_soft = loss_soft_fn(student_preds, teacher_preds)
                loss = 0.4 * loss_hard + 0.6 * loss_soft

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(targets_mt)
                n_samples += len(targets_mt)

            scheduler.step()
            avg_loss = running_loss / n_samples

            student.eval()
            val_preds_all = []
            val_tgt_all = []
            with torch.no_grad():
                for spectra, targets_mt in val_loader:
                    spectra = spectra.to(device)
                    preds_norm = student(spectra).cpu()
                    preds_mt = preds_norm * TARGET_STD + TARGET_MEAN
                    val_preds_all.append(preds_mt)
                    val_tgt_all.append(targets_mt)
            val_preds_mt = torch.cat(val_preds_all).numpy()
            val_tgt_mt = torch.cat(val_tgt_all).numpy()
            val_mae_ut = float(np.abs(val_preds_mt - val_tgt_mt).mean()) * 1000

            if val_mae_ut < best_val_mae:
                best_val_mae = val_mae_ut
                torch.save(student.state_dict(), student_ckpt)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:3d}/{EPOCHS} | loss={avg_loss:.5f} | "
                      f"val={val_mae_ut:.1f} μT | best={best_val_mae:.1f} μT")

        print(f"\n  Training done. Best val MAE = {best_val_mae:.1f} μT")

    # ── Evaluate best checkpoint on test set ──
    student_eval = TinyResNet1D().to(device)
    student_eval.load_state_dict(
        torch.load(student_ckpt, map_location=device, weights_only=True)
    )
    student_eval.eval()

    mae_ut, max_err_ut, r2 = evaluate(student_eval, test_spec, test_targets, device)
    print(f"  Test: MAE={mae_ut:.1f} μT | MaxErr={max_err_ut:.1f} μT | R²={r2:.4f}")

    # ── Export to ONNX ──
    onnx_path = ONNX_DIR / "v2_student_fp32.onnx"
    student_cpu = student_eval.cpu().eval()
    dummy = torch.randn(1, 512)
    torch.onnx.export(
        student_cpu, dummy, str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  ONNX exported -> {onnx_path}  ({size_kb:.1f} KB)")

    # ── Latency (ONNX) ──
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    lat_ms = _onnx_latency(sess)

    s = score_s(mae_ut, size_kb, lat_ms)
    print(f"\n  V2 Distilled Student — MAE={mae_ut:.1f} μT | Size={size_kb:.1f} KB | "
          f"Latency={lat_ms:.3f} ms | S={s:.2f}")

    # ── Write to benchmark ──
    write_benchmark_row(dict(
        tag="V2 Distilled Student",
        mae_ut=f"{mae_ut:.1f}",
        max_err_ut=f"{max_err_ut:.1f}",
        r2=f"{r2:.4f}",
        params=str(n_params),
        size_kb=f"{size_kb:.1f}",
        latency_ms=f"{lat_ms:.3f}",
        score=f"{s:.2f}",
        notes=f"TinyResNet1D base_ch=8, KD alpha=0.6",
    ))

    print()
    print_benchmark_table()


# ═════════════════════════════════════════════════════════════════════
# VARIANT 3 — Distilled Student + PTQ INT8
# ═════════════════════════════════════════════════════════════════════

def variant3_student_int8(test_spec, test_targets):
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("\n" + "=" * 60)
    print("VARIANT 3: Student + PTQ INT8")
    print("=" * 60)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    student_ckpt = CKPT_DIR / "v2_student_best.pt"
    fp32_path = ONNX_DIR / "v2_student_fp32.onnx"
    int8_path = ONNX_DIR / "v3_student_int8.onnx"

    if not fp32_path.exists():
        print("  Exporting student FP32 ONNX ...")
        student = TinyResNet1D()
        student.load_state_dict(
            torch.load(student_ckpt, map_location="cpu", weights_only=True)
        )
        student.eval()
        dummy = torch.randn(1, 512)
        torch.onnx.export(
            student, dummy, str(fp32_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
    print(f"  FP32 ONNX: {fp32_path}  ({os.path.getsize(fp32_path)/1024:.1f} KB)")

    quantize_dynamic(
        str(fp32_path), str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    int8_kb = os.path.getsize(int8_path) / 1024
    print(f"  Quantized INT8: {int8_path}  ({int8_kb:.1f} KB)")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(int8_path), opts, providers=["CPUExecutionProvider"])

    spec_np = test_spec.numpy()
    tgt_np = test_targets.numpy()
    mae_ut, max_err_ut, r2 = _onnx_evaluate(sess, spec_np, tgt_np)
    lat_ms = _onnx_latency(sess)
    s = score_s(mae_ut, int8_kb, lat_ms)

    print(f"\n  V3 Student+INT8 — MAE={mae_ut:.1f} μT | Size={int8_kb:.1f} KB | "
          f"Latency={lat_ms:.3f} ms | S={s:.2f}")

    write_benchmark_row(dict(
        tag="V3 Student+INT8",
        mae_ut=f"{mae_ut:.1f}",
        max_err_ut=f"{max_err_ut:.1f}",
        r2=f"{r2:.4f}",
        params="12073",
        size_kb=f"{int8_kb:.1f}",
        latency_ms=f"{lat_ms:.3f}",
        score=f"{s:.2f}",
        notes="TinyResNet1D + ONNX dynamic QInt8",
    ))


# ═════════════════════════════════════════════════════════════════════
# VARIANT 4 — Magnitude Pruning (70% sparsity)
# ═════════════════════════════════════════════════════════════════════

def variant4_pruning(test_spec, test_targets, device):
    import torch.nn.utils.prune as prune
    import onnxruntime as ort
    from torch.utils.data import TensorDataset, DataLoader

    print("\n" + "=" * 60)
    print("VARIANT 4: Magnitude Pruning (70% sparsity)")
    print("=" * 60)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    pruned_ckpt = CKPT_DIR / "v4_pruned70_best.pt"

    model = _load_baseline_model().to(device)

    parameters_to_prune = [
        (m, "weight") for m in model.modules()
        if isinstance(m, (nn.Conv1d, nn.Linear))
    ]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.70,
    )
    total_w = sum(m.weight.nelement() for m, _ in parameters_to_prune)
    nonzero_w = sum((m.weight != 0).sum().item() for m, _ in parameters_to_prune)
    print(f"  Pruned: {total_w - nonzero_w}/{total_w} weights zeroed "
          f"({100*(total_w - nonzero_w)/total_w:.1f}% sparsity)")

    # ── Fine-tune ──
    train_spec, train_tgt = load_filtered("train")
    val_spec, val_tgt = load_filtered("val")
    print(f"  Train: {len(train_spec)}  Val: {len(val_spec)}")

    train_loader = DataLoader(
        TensorDataset(train_spec, train_tgt),
        batch_size=256, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_spec, val_tgt),
        batch_size=512, shuffle=False, num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.5)

    best_val_mae = float("inf")
    for epoch in range(1, 31):
        model.train()
        for spectra, targets_mt in train_loader:
            spectra = spectra.to(device)
            targets_norm = ((targets_mt.to(device)) - TARGET_MEAN) / TARGET_STD
            optimizer.zero_grad()
            loss = criterion(model(spectra), targets_norm)
            loss.backward()
            optimizer.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for spectra, targets_mt in val_loader:
                spectra = spectra.to(device)
                p = model(spectra).cpu() * TARGET_STD + TARGET_MEAN
                vp.append(p)
                vt.append(targets_mt)
        val_mae_ut = float(np.abs(
            torch.cat(vp).numpy() - torch.cat(vt).numpy()
        ).mean()) * 1000

        if val_mae_ut < best_val_mae:
            best_val_mae = val_mae_ut
            torch.save(model.state_dict(), pruned_ckpt)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:2d}/30 | val={val_mae_ut:.1f} μT | best={best_val_mae:.1f} μT")

    print(f"  Fine-tune done. Best val MAE = {best_val_mae:.1f} μT")

    # ── Load best checkpoint (still has pruning hooks) then make permanent ──
    model.load_state_dict(
        torch.load(pruned_ckpt, map_location=device, weights_only=True)
    )
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")
    model.eval()

    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    total = sum(p.numel() for p in model.parameters())
    theoretical_kb = nonzero * 4 / 1024
    print(f"  Non-zero params: {nonzero:,} / {total:,}")

    # ── Export to ONNX ──
    onnx_path = ONNX_DIR / "v4_pruned70.onnx"
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, 512)
    torch.onnx.export(
        model_cpu, dummy, str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    actual_kb = os.path.getsize(onnx_path) / 1024

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    spec_np = test_spec.numpy()
    tgt_np = test_targets.numpy()
    mae_ut, max_err_ut, r2 = _onnx_evaluate(sess, spec_np, tgt_np)
    lat_ms = _onnx_latency(sess)
    s = score_s(mae_ut, actual_kb, lat_ms)

    print(f"\n  V4 Pruned 70% — MAE={mae_ut:.1f} μT | Size={actual_kb:.1f} KB | "
          f"Latency={lat_ms:.3f} ms | S={s:.2f}")
    print(f"  Note: unstructured pruning does not reduce file size in standard formats.")
    print(f"    Actual size={actual_kb:.1f} KB. Theoretical size at 70% sparsity={theoretical_kb:.1f} KB.")

    write_benchmark_row(dict(
        tag="V4 Pruned 70%",
        mae_ut=f"{mae_ut:.1f}",
        max_err_ut=f"{max_err_ut:.1f}",
        r2=f"{r2:.4f}",
        params=f"{nonzero}",
        size_kb=f"{actual_kb:.1f}",
        latency_ms=f"{lat_ms:.3f}",
        score=f"{s:.2f}",
        notes=f"70% global L1 pruning + 30ep finetune. Theoretical={theoretical_kb:.1f}KB",
    ))


# ═════════════════════════════════════════════════════════════════════
# PARETO PLOTS
# ═════════════════════════════════════════════════════════════════════

def _tag_color(tag):
    tag_l = tag.lower()
    if "baseline" in tag_l:
        return "gray"
    if "v1" in tag_l:
        return "tab:blue"
    if "v2" in tag_l:
        return "tab:green"
    if "v3" in tag_l:
        return "tab:red"
    if "v4" in tag_l:
        return "tab:orange"
    return "black"


def _pareto_front(xs, ys):
    """Return indices on the Pareto front (lower x AND lower y are better)."""
    pts = sorted(range(len(xs)), key=lambda i: (xs[i], ys[i]))
    front = []
    best_y = float("inf")
    for i in pts:
        if ys[i] < best_y:
            front.append(i)
            best_y = ys[i]
    return front


def plot_pareto_fronts():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BENCHMARK_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    tags = [r["tag"] for r in rows]
    maes = [float(r["mae_ut"]) for r in rows]
    sizes = [float(r["size_kb"]) for r in rows]
    lats = [float(r["latency_ms"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    colors = [_tag_color(t) for t in tags]

    # ── Plot 1: MAE vs Size ──
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(len(tags)):
        ax.scatter(sizes[i], maes[i], c=colors[i], s=80, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate(tags[i], (sizes[i], maes[i]), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    front = _pareto_front(sizes, maes)
    if len(front) > 1:
        fx = [sizes[i] for i in front]
        fy = [maes[i] for i in front]
        ax.step(fx, fy, where="post", color="black", linewidth=1.5, linestyle="--",
                label="Pareto front", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (KB)")
    ax.set_ylabel("MAE (μT)")
    ax.set_title("Pareto Front: MAE vs Model Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pareto_mae_vs_size.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: MAE vs Latency ──
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(len(tags)):
        ax.scatter(lats[i], maes[i], c=colors[i], s=80, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate(tags[i], (lats[i], maes[i]), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    front_lat = _pareto_front(lats, maes)
    if len(front_lat) > 1:
        fx = [lats[i] for i in front_lat]
        fy = [maes[i] for i in front_lat]
        ax.step(fx, fy, where="post", color="black", linewidth=1.5, linestyle="--",
                label="Pareto front", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("MAE (μT)")
    ax.set_title("Pareto Front: MAE vs Latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pareto_mae_vs_latency.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Score S bar chart ──
    order = sorted(range(len(tags)), key=lambda i: scores[i], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(order))
    bar_colors = [colors[i] for i in order]
    bar_vals = [scores[i] for i in order]
    bar_labels = [tags[i] for i in order]
    bars = ax.barh(y_pos, bar_vals, color=bar_colors, edgecolor="k", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Compression Score S")
    ax.set_title("Compression Scores")
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "compression_scores.png", dpi=150)
    plt.close(fig)

    print(f"  Saved pareto_mae_vs_size.png, pareto_mae_vs_latency.png, compression_scores.png")

    # ── Summary ──
    front_tags_size = [tags[i] for i in _pareto_front(sizes, maes)]
    front_tags_lat = [tags[i] for i in _pareto_front(lats, maes)]
    all_front = sorted(set(front_tags_size + front_tags_lat))
    print(f"  Pareto-optimal variants: {all_front}")

    recommended = [tags[i] for i in range(len(tags))
                   if scores[i] > 10 and maes[i] < 700]
    if recommended:
        print(f"  Recommended deployment model: {recommended}")
    else:
        print("  No variant meets S>10 AND MAE<700 μT.")


# ═════════════════════════════════════════════════════════════════════

def main():
    device = get_device()
    print(f"Device: {device}")

    test_spec, test_targets = load_filtered("test")
    print(f"Test samples loaded: {len(test_spec)}")
    print(f"\nBaseline metrics:")
    for k, v in BASELINE.items():
        print(f"  {k}: {v}")

    variant1_ptq_int8(test_spec, test_targets)
    variant2_distillation(test_spec, test_targets, device)
    variant3_student_int8(test_spec, test_targets)
    variant4_pruning(test_spec, test_targets, device)

    print("\n" + "=" * 60)
    print("PARETO ANALYSIS")
    print("=" * 60)
    plot_pareto_fronts()

    print("\n" + "=" * 60)
    print("FINAL BENCHMARK")
    print("=" * 60)
    print_benchmark_table()


if __name__ == "__main__":
    main()
