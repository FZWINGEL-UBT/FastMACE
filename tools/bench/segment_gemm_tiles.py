"""Tile screen of the ``segment_gemm`` Triton GEMM against cuBLAS (fp32 profile, O2; 28 October 2026).

The profile found the joined products at 8.7–12.3 TFLOPS where cuBLAS reaches 13–15 TFLOPS on the same row
counts (NbCl5, NaNbO3). The kernel has one tile configuration (64 × 64 × 32, 4 warps, 3 stages, picked on
sm_89 in round 1). This times every candidate configuration on the MH-1 product shapes of the three benchmark
cells, ``[atoms (2l + 1), K] @ [K, N]``, in CUDA-graph replays (``--replays`` per graph, median of
``--repeats``), next to ``torch.mm`` (cuBLAS), and compares each output with the current configuration's (max
abs difference; 0 means bit-identical). Rows are contiguous here (D = 1); the gathered addressing of the
library is the same arithmetic. Measurement only; hold the local GPU reservation (``--quiet``). Writes
tiles.json and tiles.csv into --output.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401

CONFIGS = [  # (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages); the first is the library's
    (64, 64, 32, 4, 3), (64, 64, 32, 4, 4), (64, 64, 16, 4, 4), (128, 64, 32, 4, 3), (128, 64, 32, 8, 3),
    (64, 128, 32, 4, 3), (64, 128, 32, 8, 3), (128, 128, 32, 8, 3), (128, 128, 16, 8, 4), (128, 64, 16, 4, 4),
    (32, 64, 32, 4, 3), (64, 32, 32, 4, 3),
]
ATOMS = {"NaCl": 216, "NaNbO3": 1080, "NbCl5": 1944}
KN = [(512, 512), (512, 384), (384, 512), (512, 256), (256, 512), (512, 128), (128, 512), (2048, 384),
      (384, 2048), (2048, 256), (256, 2048)]
DEGREES = (1, 3, 5, 7)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--structures", nargs="+", default=["NaNbO3", "NbCl5"])
    args = p.parse_args(argv)
    import torch
    import triton

    from fastmace.kernels.triton import segment_gemm

    args.output.mkdir(parents=True, exist_ok=False)
    kernel = segment_gemm._kernel()
    torch.manual_seed(0)
    rows_out = []

    def timed(fn):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.replays):
                fn()
        times = []
        for _ in range(args.repeats):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            graph.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e3 / args.replays)
        return statistics.median(times)

    for structure in args.structures:
        for degree in DEGREES:
            m = ATOMS[structure] * degree
            for k, n in KN:
                if degree > 1 and k == 2048 or degree > 1 and n == 2048:
                    continue  # the 2048-wide products are l = 0 only
                a = torch.randn(m, k, device="cuda")
                b = torch.randn(k, n, device="cuda") / k ** 0.5
                c_ref = torch.empty(m, n, device="cuda")
                cublas = timed(lambda: torch.mm(a, b, out=c_ref))
                outputs = {}
                for config in CONFIGS:
                    bm, bn, bk, warps, stages = config
                    c = torch.empty(m, n, device="cuda")

                    def run():
                        kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                            a, b, c, m, n, k, 1, k, 0, b.stride(0), b.stride(1), n, 0, BLOCK_M=bm, BLOCK_N=bn,
                            BLOCK_K=bk, PRECISION="ieee", num_warps=warps, num_stages=stages)
                    try:
                        us = timed(run)
                    except Exception as error:  # noqa: BLE001  (a config that does not compile or fit)
                        rows_out.append(dict(structure=structure, m=m, k=k, n=n, config=str(config), us=None,
                                             error=str(error)[:120]))
                        continue
                    outputs[config] = c
                    diff = float((c - outputs[CONFIGS[0]]).abs().max()) if CONFIGS[0] in outputs else None
                    rows_out.append(dict(structure=structure, m=m, k=k, n=n, config=str(config), us=us,
                                         cublas_us=cublas, ratio_to_cublas=us / cublas,
                                         tflops=2 * m * n * k / us / 1e6, max_abs_diff_to_current=diff))
    keys = ["structure", "m", "k", "n", "config", "us", "cublas_us", "ratio_to_cublas", "tflops",
            "max_abs_diff_to_current", "error"]
    with (args.output / "tiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows_out)
    totals = {}
    for row in rows_out:
        if row.get("us") is not None:
            totals.setdefault(row["config"], [0.0, 0.0])
            totals[row["config"]][0] += row["us"]
            totals[row["config"]][1] += row["cublas_us"]
    summary = dict(tool="segment_gemm_tiles", created_utc=datetime.now(timezone.utc).isoformat(),
                   gpu=torch.cuda.get_device_name(), structures=args.structures,
                   total_us_by_config={c: v[0] for c, v in totals.items()},
                   cublas_total_us=next(iter(totals.values()))[1] if totals else None)
    (args.output / "tiles.json").write_text(json.dumps(summary, indent=2) + "\n")
    for config, (us, cub) in sorted(totals.items(), key=lambda item: item[1][0]):
        print(f"{config:24} sum {us:9.1f} us   ({us / cub:.3f} of cuBLAS sum {cub:.1f} us)")


if __name__ == "__main__":
    main()
