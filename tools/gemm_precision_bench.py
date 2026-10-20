"""FP32 GEMM precision modes on the production shapes (plan opt-1, OPT-T01 step 2).

For every GEMM operator of one force call (``gemms.csv`` of tools/step_breakdown.py: the forward products
and their input gradients, with the strides the compiled model uses) it times and checks

  cublas_ieee    torch.mm / addmm with ``torch.backends.cuda.matmul.fp32_precision = "ieee"`` (production)
  cublas_tf32    the same with "tf32" (1xTF32: not FP32-accurate; shows the tensor-core rate only)
  triton_ieee    a plain Triton GEMM, ``tl.dot(input_precision="ieee")`` (control for the Triton kernel)
  triton_tf32x3  the same kernel with ``input_precision="tf32x3"`` (3xTF32, FP32-class accuracy)

Each Triton mode runs a small fixed list of tile configurations and reports the fastest. That is a
screen's upper bound, not a production choice: an adopted kernel would need one configuration by a
shape rule (no autotuning, SPEC §9.1). Errors are against a float64 product of the same float32 inputs:
the largest absolute error over the largest |reference|, and the relative Frobenius error, with the
ratio to cuBLAS IEEE's error. A peak test at 4096^3 compares the rates of the modes on the card, and a
capture test checks that a tf32x3 kernel replays bitwise in a CUDA graph.

Use an otherwise idle GPU and hold the local reservation (tools/ci/local_gpu.py run). Output:
<output>/summary.json and shapes.csv. Measurement only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

MODES = ("cublas_ieee", "cublas_tf32", "triton_ieee", "triton_tf32x3")
TRITON_PRECISION = {"triton_ieee": "ieee", "triton_tf32x3": "tf32x3"}
# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages): a fixed list for the screen.
CONFIGS = ((64, 64, 32, 4, 3), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3), (128, 128, 32, 8, 3),
           (32, 64, 32, 4, 3), (64, 64, 16, 4, 3), (32, 32, 16, 2, 3))


def _shapes(text):
    try:
        value = json.loads(text) if text else []
    except json.JSONDecodeError:
        return []
    return [s for s in value if isinstance(s, list)]


def operands(op, dims, strides):
    """(M, K, N, A strides, B strides) of a GEMM operator from the profiler's input dims and strides, the
    same reading as tools/step_breakdown.gemm_flops (out= variants, addmm's 1-D or 2-D bias); else None."""
    shapes, stride_list = _shapes(dims), _shapes(strides)
    stride_list = stride_list + [None] * (len(shapes) - len(stride_list))
    two = [(s, t) for s, t in zip(shapes, stride_list) if len(s) == 2 and all(isinstance(v, int) for v in s)]
    if op not in ("aten::mm", "aten::addmm") or len(two) < 2:
        return None
    first = 0
    if (op == "aten::addmm" and len(two) >= 3 and two[1][0][1] == two[2][0][0]
            and two[0][0] == [two[1][0][0], two[2][0][1]]):
        first = 1
    (a, a_stride), (b, b_stride) = two[first], two[first + 1]
    if a[1] != b[0]:
        return None
    return a[0], a[1], b[1], (tuple(a_stride) if a_stride else None), (tuple(b_stride) if b_stride else None)


def read_gemms(path):
    """Production GEMMs of gemms.csv: one entry per operator row, with M, K, N, strides and calls per force
    call."""
    out = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = operands(row["op"], row["input_dims"], row.get("input_strides", ""))
            if parsed is None:
                continue
            m, k, n, a_stride, b_stride = parsed
            out.append(dict(category=row["category"], op=row["op"], m=m, k=k, n=n, a_stride=a_stride,
                            b_stride=b_stride, invocations_per_call=float(row["invocations_per_call"]),
                            production_us_per_call=float(row["us_per_call"])))
    return out


def relative_errors(got, reference):
    """(max |got - ref| / max |ref|, ||got - ref||_F / ||ref||_F) of two float64 arrays or tensors."""
    diff = got - reference
    scale = float(abs(reference).max())
    frob = float((reference ** 2).sum()) ** 0.5
    return (float(abs(diff).max()) / scale if scale else 0.0,
            float((diff ** 2).sum()) ** 0.5 / frob if frob else 0.0)


def _strided(torch, shape, stride, generator):
    """A float32 CUDA tensor of ``shape`` with ``stride`` (a transposed view when the stride says so)."""
    rows, cols = shape
    if stride is None or tuple(stride) == (cols, 1):
        return torch.randn(rows, cols, generator=generator, device="cuda")
    if tuple(stride) == (1, rows):
        return torch.randn(cols, rows, generator=generator, device="cuda").t()
    storage = torch.randn((rows - 1) * stride[0] + (cols - 1) * stride[1] + 1, generator=generator, device="cuda")
    return storage.as_strided((rows, cols), tuple(stride))


def triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        pm, pn = tl.program_id(0), tl.program_id(1)
        rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            kk = start + rk
            a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak,
                        mask=(rm < M)[:, None] & (kk < K)[None, :], other=0.0)
            b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn,
                        mask=(kk < K)[:, None] & (rn < N)[None, :], other=0.0)
            acc += tl.dot(a, b, input_precision=PRECISION)
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm < M)[:, None] & (rn < N)[None, :])

    return gemm


def run_triton(kernel, a, b, out, config, precision):
    import triton
    bm, bn, bk, warps, stages = config
    m, k = a.shape
    n = b.shape[1]
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    kernel[grid](a, b, out, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0),
                 out.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, PRECISION=precision, num_warps=warps,
                 num_stages=stages)


def time_us(torch, fn, repeats):
    """Median µs of ``repeats`` calls, each between CUDA events, after warm-up."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1e3)
    return statistics.median(times)


def measure(torch, kernel, a, b, repeats):
    """{mode: dict(us, config, max_rel, frob_rel)} for one product a @ b."""
    reference = a.double() @ b.double()
    out = torch.empty(a.shape[0], b.shape[1], device="cuda")
    results = {}
    for mode in ("cublas_ieee", "cublas_tf32"):
        torch.backends.cuda.matmul.fp32_precision = "ieee" if mode == "cublas_ieee" else "tf32"
        got = a @ b
        us = time_us(torch, lambda: torch.mm(a, b, out=out), repeats)
        results[mode] = dict(us=us, config=None, errors=relative_errors(got.double(), reference))
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    for mode, precision in TRITON_PRECISION.items():
        best = None
        for config in CONFIGS:
            try:
                run_triton(kernel, a, b, out, config, precision)
            except Exception as exc:  # a configuration the compiler refuses (shared memory, ...)
                results.setdefault(f"{mode}_refused", []).append(f"{config}: {type(exc).__name__}")
                continue
            us = time_us(torch, lambda c=config: run_triton(kernel, a, b, out, c, precision), repeats)
            if best is None or us < best[0]:
                best = (us, config, relative_errors(out.double(), reference))
        if best is not None:
            results[mode] = dict(us=best[0], config=list(best[1]), errors=best[2])
    return results


def capture_check(torch, kernel):
    """A tf32x3 kernel in a CUDA graph: its replay equals the eager result bitwise."""
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    out = torch.empty(512, 512, device="cuda")
    config = CONFIGS[0]
    run_triton(kernel, a, b, out, config, "tf32x3")
    torch.cuda.synchronize()
    eager = out.clone()
    out.zero_()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            run_triton(kernel, a, b, out, config, "tf32x3")
    graph.replay()
    torch.cuda.synchronize()
    return dict(bitwise_equal=bool(torch.equal(out, eager)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--gemms", type=Path, nargs="+", default=[],
                   help="gemms.csv files of tools/step_breakdown.py (one per structure)")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--peak-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=20261020)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ab_step_time  # noqa: E402  (tools/: commit, GPU processes)

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    kernel = triton_kernel()
    import triton

    summary = dict(tool="gemm_precision_bench", created_utc=datetime.now(timezone.utc).isoformat(),
                   host=os.uname().nodename, fastmace_commit=ab_step_time.fastmace_commit(),
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                   cuda=torch.version.cuda, configs=[list(c) for c in CONFIGS], repeats=args.repeats)
    n = args.peak_size
    a = torch.randn(n, n, generator=generator, device="cuda")
    b = torch.randn(n, n, generator=generator, device="cuda")
    peak = measure(torch, kernel, a, b, max(10, args.repeats // 5))
    summary["peak"] = {mode: dict(r, tflops=2 * n ** 3 / (r["us"] * 1e-6) / 1e12) for mode, r in peak.items()
                       if isinstance(r, dict)}
    del a, b
    summary["capture"] = capture_check(torch, kernel)
    rows = []
    for path in args.gemms:
        structure = path.parent.name
        for gemm in read_gemms(path):
            a = _strided(torch, (gemm["m"], gemm["k"]), gemm["a_stride"], generator)
            b = _strided(torch, (gemm["k"], gemm["n"]), gemm["b_stride"], generator)
            results = measure(torch, kernel, a, b, args.repeats)
            row = dict(structure=structure, **{k: v for k, v in gemm.items() if not k.endswith("stride")},
                       a_stride=str(gemm["a_stride"]), b_stride=str(gemm["b_stride"]))
            for mode in MODES:
                r = results.get(mode)
                if r is None:
                    continue
                row[f"{mode}_us"] = r["us"]
                row[f"{mode}_tflops"] = 2 * gemm["m"] * gemm["k"] * gemm["n"] / (r["us"] * 1e-6) / 1e12
                row[f"{mode}_max_rel"], row[f"{mode}_frob_rel"] = r["errors"]
                row[f"{mode}_config"] = r["config"]
            row["refused"] = {k: v for k, v in results.items() if k.endswith("_refused")}
            rows.append(row)
            print(structure, gemm["m"], gemm["k"], gemm["n"],
                  " ".join(f"{m} {row.get(f'{m}_us', float('nan')):.1f}us" for m in MODES), flush=True)
    per_structure = {}
    for row in rows:
        totals = per_structure.setdefault(row["structure"], {m: 0.0 for m in MODES} | {"production_profiled": 0.0})
        totals["production_profiled"] += row["production_us_per_call"]
        for mode in MODES:
            if f"{mode}_us" in row:
                totals[mode] += row[f"{mode}_us"] * row["invocations_per_call"]
    summary["gemm_us_per_force_call"] = per_structure
    summary["worst_error"] = {mode: max((row.get(f"{mode}_max_rel", 0.0) for row in rows), default=None)
                              for mode in MODES}
    summary["tf32x3_faster_than_cublas_ieee"] = sum(
        1 for row in rows if row.get("triton_tf32x3_us", float("inf")) < row["cublas_ieee_us"])
    summary["shapes"] = len(rows)
    fields = sorted({key for row in rows for key in row}, key=lambda k: (not k[0].isalpha(), k))
    with (args.output / "shapes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("peak", "capture", "gemm_us_per_force_call", "worst_error",
                                              "tf32x3_faster_than_cublas_ieee", "shapes")}, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
