#!/usr/bin/env python3
"""Standalone row/col-normalized 3xFP16 GEMM screen for sm_80/sm_89.

Times cuBLAS IEEE FP32 against the Triton 3xFP16 kernel of OPT5-T06/T08 on the
production GEMM geometries recorded in gemms.csv (NaCl and NbCl5 force calls).
This is a falsification screen, not a production dispatch: the FP16x3 time is
the best of a small fixed tile list, so a win only earns a shape-rule design
and a force gate; a loss closes the standalone path on that GPU.

Output: <output>/summary.json (the output directory must not exist).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemm_precision_bench as gpb  # noqa: E402
import opt5_t08_hybrid_gemm as hybrid  # noqa: E402

CONFIGS = ((64, 64, 32, 4, 3), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3), (128, 128, 32, 8, 3))


def source_commit() -> str | None:
    marker = Path(".source_commit")
    return marker.read_text().strip() if marker.is_file() else None


def aggregate(paths):
    """Unique operand geometries with total invocations per force call, per structure."""
    structures = {}
    for path in paths:
        grouped = {}
        for row in gpb.read_gemms(path):
            key = (row["m"], row["k"], row["n"], str(row["a_stride"]), str(row["b_stride"]))
            entry = grouped.setdefault(key, dict(row))
            entry["invocations_per_call"] = entry.get("_calls", 0.0) + row["invocations_per_call"]
            entry["_calls"] = entry["invocations_per_call"]
        structures[path.parent.name] = list(grouped.values())
    return structures


def view(torch, row, generator, left):
    """The production-strided operand of one gemms.csv row (left = A [M,K], right = B [K,N])."""
    shape = (row["m"], row["k"]) if left else (row["k"], row["n"])
    stride = row["a_stride"] if left else row["b_stride"]
    if isinstance(stride, str):
        try:
            stride = ast.literal_eval(stride)
        except (ValueError, SyntaxError):
            stride = None
    if not isinstance(stride, (list, tuple)):
        stride = None
    return gpb._strided(torch, shape, tuple(stride) if stride else None, generator)


def time_config(a, b, out, config):
    bm, bn, bk, nw, ns = config
    return hybrid.graph_time_us(
        lambda: hybrid.launch_triton_fp16x3_rowcol(a, b, out, bm, bn, bk, nw, ns), reps=30, blocks=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gemms", type=Path, nargs="+", default=[
        Path("results/2-T20a-step-breakdown-20261020/NaCl/gemms.csv"),
        Path("results/2-T20a-step-breakdown-20261020/NbCl5/gemms.csv")])
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    import torch
    import triton

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    except Exception:
        pass

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    summary = dict(tool="fp16x3_gemm_screen", created_utc=datetime.now(timezone.utc).isoformat(),
                   gpu=torch.cuda.get_device_name(),
                   capability=".".join(map(str, torch.cuda.get_device_capability())),
                   torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
                   source_commit=source_commit(), configs=[list(c) for c in CONFIGS],
                   mode="best-of-fixed-tiles screen (not a production shape rule)", structures={})

    for structure, rows in aggregate(args.gemms).items():
        totals = {"cublas_ieee": 0.0, "triton_fp16x3_rowcol": 0.0}
        worst = {"cublas_ieee": 0.0, "triton_fp16x3_rowcol": 0.0}
        measured = []
        for row in rows:
            a = view(torch, row, generator, left=True)
            b = view(torch, row, generator, left=False)
            out = torch.empty((row["m"], row["n"]), device="cuda", dtype=torch.float32)
            ref64 = a.double() @ b.double()
            abs64 = a.double().abs() @ b.double().abs()

            t_cublas = hybrid.graph_time_us(lambda: torch.mm(a, b, out=out), reps=30, blocks=5)
            err_cublas = hybrid.calc_errors(torch.mm(a, b), ref64, abs64)

            best = None
            refused = []
            for config in CONFIGS:
                try:
                    us = time_config(a, b, out, config)
                except Exception as exc:  # a tile the compiler refuses on this GPU
                    refused.append(f"{config}: {type(exc).__name__}")
                    continue
                if best is None or us < best[0]:
                    best = (us, config)
            if best is None:
                raise RuntimeError(f"Every FP16x3 tile refused for {row['m']}x{row['k']}x{row['n']}: {refused}")
            t_fp16, best_config = best
            hybrid.launch_triton_fp16x3_rowcol(a, b, out, *best_config)
            err_fp16 = hybrid.calc_errors(out, ref64, abs64)

            calls = float(row["invocations_per_call"])
            totals["cublas_ieee"] += t_cublas * calls
            totals["triton_fp16x3_rowcol"] += t_fp16 * calls
            worst["cublas_ieee"] = max(worst["cublas_ieee"], err_cublas["cancel_scaled"])
            worst["triton_fp16x3_rowcol"] = max(worst["triton_fp16x3_rowcol"], err_fp16["cancel_scaled"])
            measured.append(dict(m=row["m"], k=row["k"], n=row["n"], op=row["op"], category=row["category"],
                                 invocations_per_call=calls, cublas_ieee_us=t_cublas,
                                 triton_fp16x3_rowcol_us=t_fp16, best_config=list(best_config), refused=refused,
                                 errors={"cublas_ieee": err_cublas, "triton_fp16x3_rowcol": err_fp16}))
            a = b = out = None
            del ref64, abs64

        ratio = totals["triton_fp16x3_rowcol"] / totals["cublas_ieee"]
        cancel_ratio = (worst["triton_fp16x3_rowcol"] / worst["cublas_ieee"]) if worst["cublas_ieee"] else None
        summary["structures"][structure] = dict(
            unique_geometries=len(rows), weighted_us_per_force_call=totals,
            fp16x3_over_cublas_weighted=ratio, worst_cancel_scaled=worst,
            worst_cancel_ratio_fp16x3_over_ieee=cancel_ratio, rows=measured)
        print(f"{structure}: cuBLAS {totals['cublas_ieee']:.1f} us/call, "
              f"FP16x3 {totals['triton_fp16x3_rowcol']:.1f} us/call, ratio {ratio:.3f}, "
              f"worst-cancel ratio {cancel_ratio}", flush=True)

    ratios = [entry["fp16x3_over_cublas_weighted"] for entry in summary["structures"].values()]
    summary["verdict"] = dict(
        stop_rule="continue only if FP16x3 <= 0.90x cuBLAS weighted per force call on every structure and "
                  "worst cancellation-scaled error <= 2x IEEE",
        passes=bool(ratios) and all(r <= 0.90 for r in ratios) and all(
            (entry["worst_cancel_ratio_fp16x3_over_ieee"] or 999) <= 2.0
            for entry in summary["structures"].values()))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
