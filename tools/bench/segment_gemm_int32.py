"""32-bit address arithmetic in the segment_gemm Triton kernel (fp32 profile follow-up A2, 29 October 2026).

The library kernel (``fastmace.kernels.triton.segment_gemm``) computes rows, columns and gathered addresses in
int64. Every offset of the benchmark cells fits int32 by a wide margin, so this times a copy of the kernel with
int32 arithmetic against the library kernel, same tile (64 x 64 x 32, 4 warps, 3 stages), on the MH-1 product
shapes of NaNbO3 and NbCl5, in CUDA-graph replays; the outputs must be bitwise equal. Stop rule: at least 10 % on
the summed product time, else the int64 path stays. Measurement only; idle GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401

ATOMS = {"NaCl": 216, "NaNbO3": 1080, "NbCl5": 1944}
KN = [(512, 512), (512, 384), (384, 512), (512, 256), (256, 512), (512, 128), (128, 512), (2048, 384), (384, 2048)]
DEGREES = (1, 3, 5, 7)


def int32_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(a_ptr, b_ptr, c_ptr, rows, N, K, D, a_n, a_m, b_k, b_n, c_n, c_m,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        pm, pn = tl.program_id(0), tl.program_id(1)
        r = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        j = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        atom, comp = r // D, r % D
        a_row = atom * a_n + comp * a_m
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            k = start + kk
            a = tl.load(a_ptr + a_row[:, None] + k[None, :], mask=(r < rows)[:, None] & (k < K)[None, :], other=0.0)
            b = tl.load(b_ptr + k[:, None] * b_k + j[None, :] * b_n, mask=(k < K)[:, None] & (j < N)[None, :],
                        other=0.0)
            acc += tl.dot(a, b, input_precision=PRECISION)
        tl.store(c_ptr + (atom * c_n + comp * c_m)[:, None] + j[None, :], acc,
                 mask=(r < rows)[:, None] & (j < N)[None, :])

    return gemm


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--structures", nargs="+", default=["NaNbO3", "NbCl5"])
    args = p.parse_args(argv)
    import torch
    import triton

    from fastmace.kernels.triton import segment_gemm, tile_rule

    args.output.mkdir(parents=True, exist_ok=False)
    kernels = {"int64": segment_gemm._kernel(), "int32": int32_kernel()}
    bm, bn, bk, warps, stages = (tile_rule.BLOCK_M, tile_rule.BLOCK_N, tile_rule.BLOCK_K, tile_rule.NUM_WARPS,
                                 tile_rule.NUM_STAGES)

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

    torch.manual_seed(0)
    rows_out, totals, identical = [], {"int64": 0.0, "int32": 0.0}, True
    for structure in args.structures:
        for degree in DEGREES:
            for k, n in KN:
                if degree > 1 and 2048 in (k, n):
                    continue
                m = ATOMS[structure] * degree
                a = torch.randn(m, k, device="cuda")
                b = torch.randn(k, n, device="cuda") / k ** 0.5
                outs, entry = {}, dict(structure=structure, m=m, k=k, n=n)
                for name, kernel in kernels.items():
                    c = torch.empty(m, n, device="cuda")

                    def run(kernel=kernel, c=c):
                        kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                            a, b, c, m, n, k, 1, k, 0, b.stride(0), b.stride(1), n, 0, BLOCK_M=bm, BLOCK_N=bn,
                            BLOCK_K=bk, PRECISION="ieee", num_warps=warps, num_stages=stages)
                    entry[f"{name}_us"] = timed(run)
                    totals[name] += entry[f"{name}_us"]
                    outs[name] = c
                entry["bitwise_equal"] = bool(torch.equal(outs["int64"], outs["int32"]))
                identical &= entry["bitwise_equal"]
                rows_out.append(entry)
    summary = dict(tool="segment_gemm_int32", created_utc=datetime.now(timezone.utc).isoformat(),
                   gpu=torch.cuda.get_device_name(), totals_us=totals, int32_over_int64=totals["int32"] / totals["int64"],
                   bitwise_equal=identical, shapes=rows_out)
    (args.output / "int32.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['gpu']}: int64 {totals['int64']:.1f} us, int32 {totals['int32']:.1f} us, "
          f"ratio {summary['int32_over_int64']:.4f}, bitwise equal {identical}")


if __name__ == "__main__":
    main()
