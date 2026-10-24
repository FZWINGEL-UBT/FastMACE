"""3xTF32 on the real operands of one force call (plan opt-4, OPT4-T03 stage 1).

For each structure it builds ``fm.ForceField`` with ``RewriteConfig(tf32x3_gemms="on")`` (float32, the compiled
evaluation plan: no CUDA graphs, so every launch passes through Python), runs one warm-up evaluation, and then
records every GEMM the record reroutes during one more evaluation (forward products and their input gradients):

- dense products (``fastmace.kernels.triton.tf32x3_gemm``): A and B as launched;
- ``segment_gemm``'s joined products: A's gathered rows and C's scattered rows rebuilt from their strides.

Each product's tf32x3 result (the launch's own output) and cuBLAS IEEE's product of the same float32 operands are
compared with a float64 product: max relative (max |c - c64| / max |c64|), relative Frobenius and the
cancellation-scaled error max |c - c64| / sum_k |a_ik b_kj| (``docs/validation/opt-T01-tensor-cores.md`` §6-§7),
plus how much the real operands cancel (the median and largest sum |a b| / |c64|). The cancellation-scaled error
counts only outputs whose sum |a b| is at least ``SCALE_FLOOR`` (2^-103: below it one float32 unit roundoff of the
sum is subnormal, and both modes lose digits to underflow, not to the split); the unfiltered value and the sum
|a b| at its worst output are recorded beside it. A profiler pass of a third
evaluation lists the matrix products still on cuBLAS (``aten::mm``/``addmm``/``bmm``), to show which GEMMs the
record leaves alone (the edge-level radial networks). Output: ``<output>/<structure>.json`` and
``summary.json``. Measurement only; any TF32 GPU (compute capability >= 8.0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parent
SCALE_FLOOR = 2.0 ** -103  # float32's smallest normal (2^-126) over its unit roundoff (2^-23)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--model", default="mh1-r2scan", help="Label (its file under FASTMACE_MODELS)")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"],
                   help="Benchmark names (under FASTMACE_STRUCTURES) or PATH@NXxNYxNZ; 3x3x3 by default")
    p.add_argument("--rattle-A", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261024)
    return p.parse_args(argv)


def errors(torch, got, reference, absolute):
    """(max relative, relative Frobenius, cancellation-scaled max over sums >= SCALE_FLOOR, the same over every
    nonzero sum, the sum |a b| at that unfiltered worst output) of ``got`` against the float64 ``reference``;
    ``absolute`` is sum_k |a_ik b_kj|."""
    diff = (got.double() - reference).abs()
    scale = float(reference.abs().max())
    frob = float(reference.norm())
    normal = absolute >= SCALE_FLOOR
    cancel = float((diff[normal] / absolute[normal]).max()) if bool(normal.any()) else 0.0
    positive = absolute > 0
    scaled = torch.where(positive, diff / absolute.clamp_min(torch.finfo(torch.float64).tiny), 0.0)
    worst = int(scaled.argmax())
    return (float(diff.max()) / scale if scale else 0.0, float(diff.norm()) / frob if frob else 0.0, cancel,
            float(scaled.reshape(-1)[worst]), float(absolute.reshape(-1)[worst]))


class Recorder:
    """Hooks on the record's two launch functions; measures each product while ``active``."""

    def __init__(self, torch):
        self.torch = torch
        self.active = False
        self.rows = []

    def measure(self, kind, a, b, c):
        torch = self.torch
        a64, b64 = a.double(), b.double()
        reference = a64 @ b64
        absolute = a64.abs() @ b64.abs()
        ieee = a @ b
        ratio = absolute / reference.abs().clamp_min(torch.finfo(torch.float64).tiny)
        row = dict(kind=kind, m=int(a.shape[0]), k=int(a.shape[1]), n=int(b.shape[1]),
                   tf32x3=errors(torch, c, reference, absolute), ieee=errors(torch, ieee, reference, absolute),
                   cancellation_median=float(ratio.median()), cancellation_max=float(ratio.max()),
                   a_absmax=float(a.abs().max()), b_absmax=float(b.abs().max()))
        self.rows.append(row)

    def install(self):
        from fastmace.kernels.triton import segment_gemm, tf32x3_gemm

        dense, joined = tf32x3_gemm._launch, segment_gemm._gemm
        recorder = self

        def launch(a, b, out):
            dense(a, b, out)
            if recorder.active:
                recorder.measure("dense", a, b, out)

        def gemm(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision="ieee"):
            joined(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision)
            if recorder.active and precision == "tf32x3":
                k, n = b.shape
                atoms = rows // dim
                left = a.reshape(-1)[a_offset:].as_strided((atoms, dim, k), (a_n, a_m, 1)).reshape(rows, k)
                out = c.reshape(-1)[c_offset:].as_strided((atoms, dim, n), (c_n, c_m, 1)).reshape(rows, n)
                recorder.measure("joined", left, b, out)

        tf32x3_gemm._launch, segment_gemm._gemm = launch, gemm


def cublas_products(torch, ff, atoms):
    """{(op, input shapes): count} of the matrix products left to cuBLAS in one evaluation."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        ff.evaluate(atoms)
        torch.cuda.synchronize()
    found = Counter()
    for event in prof.events():
        if event.name in ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm"):
            found[(event.name, str(event.input_shapes))] += 1
    names = Counter(event.name for event in prof.events() if "tf32x3" in event.name or "joined_products" in event.name
                    or event.name.startswith("gemm"))
    return [dict(op=op, shapes=shapes, count=count) for (op, shapes), count in sorted(found.items())], dict(names)


def worst(rows, mode, index):
    return max((row[mode][index] for row in rows), default=None)


def summarise(rows):
    import statistics

    out = dict(products=len(rows), shapes=len({(r["kind"], r["m"], r["k"], r["n"]) for r in rows}))
    for index, name in enumerate(("max_rel", "frob_rel", "cancellation_scaled", "cancellation_scaled_unfiltered")):
        out[name] = {mode: worst(rows, mode, index) for mode in ("tf32x3", "ieee")}
        ratios = [row["tf32x3"][index] / row["ieee"][index] for row in rows if row["ieee"][index] > 0]
        out[name]["tf32x3_over_ieee_median"] = statistics.median(ratios) if ratios else None
        out[name]["tf32x3_over_ieee_max"] = max(ratios) if ratios else None
        out[name]["tf32x3_worse_count"] = sum(1 for row in rows if row["tf32x3"][index] > row["ieee"][index])
    out["cancellation_median_max"] = max((r["cancellation_median"] for r in rows), default=None)
    out["cancellation_max"] = max((r["cancellation_max"] for r in rows), default=None)
    return out


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch

    import fastmace as fm
    from fastmace.config import RewriteConfig

    sys.path.insert(0, str(TOOLS))
    import ab_step_time  # noqa: E402  (tools/: commit, GPU processes, structures)

    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    recorder = Recorder(torch)
    recorder.install()
    model = fm.load(args.model)
    summary = dict(tool="tf32x3_accuracy", created_utc=datetime.now(timezone.utc).isoformat(),
                   host=os.uname().nodename, fastmace_commit=ab_step_time.fastmace_commit(), model=args.model,
                   gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                   torch=torch.__version__, cuda=torch.version.cuda, rattle_A=args.rattle_A, seed=args.seed,
                   structures={})
    try:
        import triton

        summary["triton"] = triton.__version__
    except ImportError:
        pass
    for spec in args.structures:
        name, atoms = ab_step_time.read_structure(spec, None)
        rng = np.random.default_rng(args.seed)
        atoms.positions += rng.uniform(-args.rattle_A, args.rattle_A, atoms.positions.shape)
        ff = fm.ForceField(model, atoms, precision="float32", rewrites=RewriteConfig(tf32x3_gemms="on"))
        plan = ff.evaluation_plan(atoms)
        decision = plan.decision("RewriteConfig.tf32x3_gemms")
        ff.evaluate(atoms)  # warm-up: compiles
        recorder.rows = []
        recorder.active = True
        ff.evaluate(atoms)
        torch.cuda.synchronize()
        recorder.active = False
        rows = recorder.rows
        left, launches = cublas_products(torch, ff, atoms)
        record = dict(structure=str(name), natoms=len(atoms), decision=list(decision), summary=summarise(rows),
                      products=rows, cublas_products_left=left, rerouted_events=launches)
        stem = Path(str(name)).stem
        (args.output / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        summary["structures"][stem] = dict(natoms=len(atoms), decision=list(decision), **record["summary"],
                                           cublas_products_left=left)
        print(stem, json.dumps(record["summary"], default=str), flush=True)
        del ff
        torch.cuda.empty_cache()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


if __name__ == "__main__":
    main()
