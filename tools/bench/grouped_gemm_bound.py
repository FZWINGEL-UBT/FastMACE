"""Bound of the H200 ideas plan's idea 6 (one grouped launch per BlockLinear): a CUDA graph of a BlockLinear's per-l
3xTF32 products launched one by one (M = 5,000 x (2l + 1) rows for l = 0 to 3, OMAT-0 on SrTiO3 10x10x10) against one
launch over the concatenated rows (sum of M, the same K and N): what a grouped kernel with a perfect tile schedule
could save per BlockLinear. Alternating A/B rounds in one process. Keep the idea at >= 15 us per force call
(about 11 such groups per call). Where the products run in CUTLASS, also the bound of idea 8 (PDL): the same four
launches with programmatic stream serialization and no ``griddepcontrol.wait`` (unsafe, timing only); drop that idea
below 3 us per group.

    python tools/bench/grouped_gemm_bound.py [--atoms 5000] [--output FILE]
"""

import argparse
import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (soft file locks before torch)


def graph_of(fn):
    import torch
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def replay_us(graph, reps):
    import torch
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        graph.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000 / reps


def pdl_mm(a, b, out):
    """``tf32x3.mm_into`` launched with ``CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`` (timing only)."""
    import ctypes

    import torch
    from cuda.bindings import driver

    from fastmace.kernels.cutlass import tf32x3
    from fastmace.kernels.oeq_streaming.runtime import _check
    kernels = tf32x3._kernels(a.device.index if a.device.index is not None else torch.cuda.current_device())
    m, k = a.shape
    n = b.shape[1]
    _, lda = tf32x3._layout(a)
    lb, ldb = tf32x3._layout(b)
    _, ldo = tf32x3._layout(out)
    key = ("r" + lb, tf32x3.TILE)
    p, i = ctypes.c_void_p, ctypes.c_int
    values = (m, n, k, a.data_ptr(), lda, b.data_ptr(), ldb, out.data_ptr(), ldo)
    tile_m, tile_n = kernels.tile[tf32x3.TILE]
    config = driver.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = -(-m // tile_m), -(-n // tile_n), 1
    config.blockDimX, config.blockDimY, config.blockDimZ = kernels.threads[tf32x3.TILE], 1, 1
    config.sharedMemBytes = kernels.shared[key]
    config.hStream = driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    attribute = driver.CUlaunchAttribute()
    attribute.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    attribute.value.programmaticStreamSerializationAllowed = 1
    config.attrs = [attribute]
    config.numAttrs = 1
    _check(driver.cuLaunchKernelEx(config, kernels.functions[key], (values, (i, i, i, p, i, p, i, p, i)), 0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=5000)
    parser.add_argument("--shapes", nargs="*", default=["128x128", "128x384", "384x128", "256x128"],
                        metavar="KxN")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import torch

    from fastmace.kernels.triton import tf32x3_gemm

    torch.manual_seed(0)
    rows = []
    ms = [args.atoms * (2 * degree + 1) for degree in range(4)]
    for shape in args.shapes:
        k, n = (int(v) for v in shape.split("x"))
        a = [torch.randn(m, k, device="cuda") for m in ms]
        b = [torch.randn(k, n, device="cuda") for _ in ms]
        out = [torch.empty(m, n, device="cuda") for m in ms]
        a_all = torch.randn(sum(ms), k, device="cuda")
        out_all = torch.empty(sum(ms), n, device="cuda")

        def separate():
            for ai, bi, oi in zip(a, b, out):
                tf32x3_gemm._launch(ai, bi, oi)

        def joined():
            tf32x3_gemm._launch(a_all, b[0], out_all)

        graphs = {"separate": graph_of(separate), "joined": graph_of(joined)}
        if tf32x3_gemm.backend(torch.device("cuda")) == "cutlass":
            from fastmace.kernels.cutlass import tf32x3
            if all(tf32x3.takes(m, k, n) for m in ms):
                def chained():
                    for ai, bi, oi in zip(a, b, out):
                        pdl_mm(ai, bi, oi)
                graphs["pdl"] = graph_of(chained)
        times = {name: [] for name in graphs}
        for r in range(args.rounds):
            order = list(graphs) if r % 2 == 0 else list(graphs)[::-1]
            for name in order:
                times[name].append(replay_us(graphs[name], args.reps))
        med = {name: sorted(t)[len(t) // 2] for name, t in times.items()}
        single = {}
        for m, ai, bi, oi in zip(ms, a, b, out):
            g = graph_of(lambda ai=ai, bi=bi, oi=oi: tf32x3_gemm._launch(ai, bi, oi))
            single[m] = round(sorted(replay_us(g, args.reps) for _ in range(5))[2], 2)
        row = dict(k=k, n=n, ms=ms, separate_us=round(med["separate"], 2), joined_us=round(med["joined"], 2),
                   saving_us=round(med["separate"] - med["joined"], 2), single_us=single)
        if "pdl" in med:
            row.update(pdl_us=round(med["pdl"], 2), pdl_saving_us=round(med["separate"] - med["pdl"], 2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = dict(gpu=torch.cuda.get_device_name(), backend=tf32x3_gemm.backend(torch.device("cuda")), rows=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
