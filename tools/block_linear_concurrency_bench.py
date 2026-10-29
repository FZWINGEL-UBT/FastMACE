"""Concurrent per-l products of one block linear (plan opt-1, OPT-T03 micro-bench; docs/design/opt-T03-small-m.md).

One MH-1 block linear (512x0e + 512x1o + 512x2e + 512x3o in and out) has four independent products
``rows_l @ W_l`` with ``rows_l`` of shape ``[atoms (2l + 1), 512]``. For NaCl (216 atoms) and NbCl5 (1,944)
3x3x3, float32, the forward (``W_l``) and the input gradient (``W_l^T``), each variant is captured once in a
CUDA graph and timed as the median of --repeats replays:

- ``sequential``: cuBLAS, one product after another (today);
- ``streams2``: cuBLAS on two graph branches (l = 0 with 3, l = 1 with 2);
- ``streams4``: cuBLAS on four graph branches;
- ``grouped``: one Triton launch for all four products (IEEE ``tl.dot``; a tile scheduler over the groups;
  64 x 64 x 32 tiles, 4 warps, 3 stages: the shared tile rule's configuration).

Every product writes its own output and nothing is accumulated across branches, so the results do not
depend on the order of execution; the bench checks that each variant equals the sequential one bitwise
(cuBLAS variants) or within 1e-5 (grouped). Output: <output>/summary.json. Use an idle GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

DIMS = (1, 3, 5, 7)
CHANNELS = 512
STRUCTURES = {"NaCl": 216, "NbCl5": 1944}


def grouped_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(a_ptr, w_ptr, c_ptr, row_start_ptr, tile_start_ptr, groups, K, N,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # A and C hold all groups' rows one after another; W holds the groups' [K, N] weights one after another.
        pid = tl.program_id(0)
        group = 0
        for g in range(1, groups):
            group += (pid >= tl.load(tile_start_ptr + g)).to(tl.int32)
        local = pid - tl.load(tile_start_ptr + group)
        tiles_n = tl.cdiv(N, BLOCK_N)
        tm, tn = local // tiles_n, local % tiles_n
        start, stop = tl.load(row_start_ptr + group), tl.load(row_start_ptr + group + 1)
        r = start + tm * BLOCK_M + tl.arange(0, BLOCK_M)
        j = tn * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        w_base = w_ptr + group.to(tl.int64) * K * N
        for k0 in range(0, K, BLOCK_K):
            k = k0 + kk
            a = tl.load(a_ptr + r.to(tl.int64)[:, None] * K + k[None, :], mask=(r < stop)[:, None] & (k < K)[None, :],
                        other=0.0)
            w = tl.load(w_base + k[:, None] * N + j[None, :], mask=(k < K)[:, None] & (j < N)[None, :], other=0.0)
            acc += tl.dot(a, w, input_precision="ieee")
        tl.store(c_ptr + r.to(tl.int64)[:, None] * N + j[None, :], acc, mask=(r < stop)[:, None] & (j < N)[None, :])

    return gemm


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=100)
    args = p.parse_args(argv)
    import torch
    import triton

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ab_step_time  # noqa: E402

    from fastmace.kernels.triton import tile_rule

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    args.output.mkdir(parents=True, exist_ok=False)
    kernel = grouped_kernel()
    g = torch.Generator(device="cuda").manual_seed(20261020)
    streams = [torch.cuda.Stream() for _ in range(3)]

    def capture(branches):
        """A CUDA graph of the branches (callables): the first on the capture stream, the rest forked."""
        for branch in branches:
            branch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        main = torch.cuda.Stream()
        with torch.cuda.stream(main), torch.cuda.graph(graph, stream=main):
            fork = torch.cuda.Event()
            fork.record(main)
            joins = []
            for stream, branch in zip(streams, branches[1:]):
                stream.wait_event(fork)
                with torch.cuda.stream(stream):
                    branch()
                join = torch.cuda.Event()
                join.record(stream)
                joins.append(join)
            branches[0]()
            for join in joins:
                main.wait_event(join)
        return graph

    def timed(graph):
        graph.replay()
        torch.cuda.synchronize()
        out = []
        for _ in range(args.repeats):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            graph.replay()
            e.record()
            e.synchronize()
            out.append(s.elapsed_time(e) * 1e3)
        return statistics.median(out)

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    choice = tile_rule.segment_gemm_choice(1, CHANNELS, CHANNELS, sms)  # the configuration (backend aside)
    summary = dict(tool="block_linear_concurrency_bench", created_utc=datetime.now(timezone.utc).isoformat(),
                   host=os.uname().nodename, gpu=torch.cuda.get_device_name(), sms=sms, torch=torch.__version__,
                   triton=triton.__version__, fastmace_commit=ab_step_time.fastmace_commit(),
                   load=list(os.getloadavg()), repeats=args.repeats, cases={})
    for name, atoms in STRUCTURES.items():
        counts = [atoms * d for d in DIMS]
        starts = [0]
        for c in counts:
            starts.append(starts[-1] + c)
        a_all = torch.randn(starts[-1], CHANNELS, generator=g, device="cuda")
        weights = torch.randn(len(DIMS), CHANNELS, CHANNELS, generator=g, device="cuda") / CHANNELS ** 0.5
        for direction in ("forward", "input_grad"):
            w = weights if direction == "forward" else weights.transpose(1, 2).contiguous()
            a = [a_all[starts[i]:starts[i + 1]] for i in range(len(DIMS))]
            outs = {v: torch.empty(starts[-1], CHANNELS, device="cuda") for v in ("sequential", "streams2", "streams4",
                                                                                 "grouped")}

            def product(variant, i):
                return lambda: torch.mm(a[i], w[i], out=outs[variant][starts[i]:starts[i + 1]])

            graphs = {
                "sequential": capture([lambda: [product("sequential", i)() for i in range(4)]]),
                "streams2": capture([lambda: (product("streams2", 0)(), product("streams2", 3)()),
                                     lambda: (product("streams2", 1)(), product("streams2", 2)())]),
                "streams4": capture([product("streams4", i) for i in range(4)]),
            }
            tile_counts = [triton.cdiv(c, choice.block_m) * triton.cdiv(CHANNELS, choice.block_n) for c in counts]
            tile_start = torch.tensor([0] + list(torch.tensor(tile_counts).cumsum(0).tolist()), device="cuda",
                                      dtype=torch.int32)
            row_start = torch.tensor(starts, device="cuda", dtype=torch.int64)

            def grouped():
                kernel[(sum(tile_counts),)](a_all, w, outs["grouped"], row_start, tile_start, len(DIMS), CHANNELS,
                                            CHANNELS, BLOCK_M=choice.block_m, BLOCK_N=choice.block_n,
                                            BLOCK_K=choice.block_k, num_warps=choice.num_warps,
                                            num_stages=choice.num_stages)

            graphs["grouped"] = capture([grouped])
            entry = {v: dict(us=timed(graph)) for v, graph in graphs.items()}
            base = entry["sequential"]["us"]
            reference = outs["sequential"]
            for v in entry:
                entry[v]["speedup"] = base / entry[v]["us"]
                diff = float((outs[v] - reference).abs().max() / reference.abs().max())
                entry[v]["max_rel_diff_vs_sequential"] = diff
            entry["flop_tflops_sequential"] = 2 * starts[-1] * CHANNELS * CHANNELS / (base * 1e-6) / 1e12
            summary["cases"][f"{name}/{direction}"] = entry
            shown = {v: (round(e["us"], 1), round(e["speedup"], 3), f"{e['max_rel_diff_vs_sequential']:.1e}")
                     for v, e in entry.items() if isinstance(e, dict)}
            print(name, direction, shown, flush=True)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["load_after"] = list(os.getloadavg())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
