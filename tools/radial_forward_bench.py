"""OPT4-T01's falsifying microbenchmark: the folded radial last layer's forward, [E, P + 1] @ [P + 1, Q].

    python tools/radial_forward_bench.py [--out results/<run-id>/radial_forward.json]

Compares, at the production layout of MH-1's interaction (P = 64 hidden columns plus the cutoff column,
Q = 1280), CUDA-graph replays of
- ``mm68``: cuBLAS on the [E, 68] operand Inductor's ``pad_mm`` builds today (production);
- ``mm64_addr``: the cheap alternative, cuBLAS at K = 64 plus a separate rank-1 update for the cutoff column;
- ``fill``: writing [E, Q] (the write floor);
- ``triton``: a write-bound Triton kernel (``_kernel`` below: all of K in one tile, the first K // 32 * 32
  columns as an IEEE FP32 block product, the cutoff column as a rank-1 update, the output block streamed out)
  at each tile configuration of ``CONFIGS``.
Each is the median of 7 blocks of 50 replays. Errors are measured against float64.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fastmace  # noqa: F401,E402  (before torch: soft file locks)
import torch  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

EDGES = {"NaCl": 8192, "NbCl5": 56320, "NaNbO3": 82688}
P, Q = 64, 1280
CONFIGS = [(bm, bn, w, s) for bm in (32, 64, 128) for bn in (64, 128, 256) for w in (4, 8) for s in (1, 2)
           if bm * bn <= 128 * 128]


BLOCK_K = 32


@triton.jit
def _kernel(a_ptr, w_ptr, out_ptr, M, N, K, sam, sak, swn, swk, som, K_MAIN: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Column tiles inner: the programs of one row block run together and share its rows of a in L2.
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    m = (pid // n_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = (pid % n_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask, col_mask = m < M, n < N
    m = m.to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K_MAIN, BLOCK_K):
        a = tl.load(a_ptr + m[:, None] * sam + (start + k)[None, :] * sak, mask=row_mask[:, None], other=0.0)
        w = tl.load(w_ptr + (start + k)[:, None] * swk + n[None, :] * swn, mask=col_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, input_precision="ieee")
    for j in range(K_MAIN, K):
        a = tl.load(a_ptr + m * sam + j * sak, mask=row_mask, other=0.0)
        w = tl.load(w_ptr + n * swn + j * swk, mask=col_mask, other=0.0)
        acc += a[:, None] * w[None, :]
    tl.store(out_ptr + m[:, None] * som + n[None, :], acc, mask=row_mask[:, None] & col_mask[None, :])


def radial_forward_into(a, weight, out, tiles):
    """out = a @ weight.T with ``_kernel`` at tiles (BLOCK_M, BLOCK_N, warps, stages)."""
    rows, k = a.shape
    cols = weight.shape[0]
    block_m, block_n, warps, stages = tiles
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(cols, block_n),)
    _kernel[grid](a, weight, out, rows, cols, k, a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
                  out.stride(0), K_MAIN=k // BLOCK_K * BLOCK_K, BLOCK_M=block_m, BLOCK_N=block_n,
                  BLOCK_K=BLOCK_K, num_warps=warps, num_stages=stages)
    return out


def timed(fn, reps=50, blocks=7):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(blocks):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / reps)
    return statistics.median(times)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cells", nargs="*", default=list(EDGES))
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    rows = []
    for cell in args.cells:
        e = EDGES[cell]
        hidden = torch.randn(e, P, device="cuda")
        cutoff = torch.rand(e, 1, device="cuda")
        a = torch.cat([hidden * cutoff, cutoff], 1).contiguous()  # [E, 65]
        weight = torch.randn(Q, P + 1, device="cuda") / 8  # folded [W, b]
        a68 = torch.nn.functional.pad(a, (0, 3)).contiguous()
        w68 = torch.nn.functional.pad(weight, (0, 3)).contiguous()
        w64, b = weight[:, :P].contiguous(), weight[:, P].contiguous()
        h64 = a[:, :P].contiguous()
        out = torch.empty(e, Q, device="cuda")
        reference = a.double() @ weight.double().t()
        row = {"cell": cell, "E": e, "K": P + 1, "N": Q}
        row["mm68_us"] = timed(lambda: torch.mm(a68, w68.t(), out=out))
        row["mm64_addr_us"] = timed(lambda: (torch.mm(h64, w64.t(), out=out), out.addr_(a[:, P], b)))
        row["fill_us"] = timed(lambda: out.fill_(1.0))
        row["write_floor_us"] = e * Q * 4 / 448e9 * 1e6  # 448 GB/s nominal (RTX 5060 Ti)
        torch.mm(a68, w68.t(), out=out)
        row["mm68_max_rel_err"] = float(((out.double() - reference).abs().max() / reference.abs().max()))
        per = {}
        for config in CONFIGS:
            try:
                per[str(config)] = timed(lambda: radial_forward_into(a, weight, out, config))
            except Exception as error:  # out of resources for a tile: record and go on
                per[str(config)] = repr(error)[:120]
        row["triton_us"] = per
        timings = {k: v for k, v in per.items() if isinstance(v, float)}
        best = min(timings, key=timings.get)
        row["triton_best"] = [best, timings[best]]
        radial_forward_into(a, weight, out, tuple(int(x) for x in best.strip("()").split(",")))
        row["triton_max_rel_err"] = float(((out.double() - reference).abs().max() / reference.abs().max()))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "triton_us"}), flush=True)
    result = {"gpu": props.name, "sms": props.multi_processor_count, "torch": torch.__version__,
              "triton": triton.__version__, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
