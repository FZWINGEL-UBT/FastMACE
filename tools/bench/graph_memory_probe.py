"""Device memory outside PyTorch's allocator around CUDA graphs (plans/capacity.md, CAP-T03).

Under WSL2 a FastMACE run holds 2-7 GiB of device memory outside PyTorch's caching allocator
(``total - free - reserved``) while its CUDA graphs live, against 0.3 GiB on native Linux for the same points
(docs/validation/capacity-T03.md). This probe isolates the cause with plain PyTorch, one case per fresh process:

- ``eager``: a large intermediate computed eagerly, then freed;
- ``graph``: the same computation captured as a CUDA graph (its intermediates in the graph's private pool), then
  replayed, then the graph deleted;
- ``graph-reuse``: like ``graph`` but the graph's output feeds a second capture in the same pool;
- ``linear``: a cuBLAS GEMM chain captured as a graph (library workspaces inside a capture);
- ``triton``: a Triton kernel captured as a graph;
- ``nodes-5000``, ``nodes-20000``: that many tiny kernels captured in one graph (memory per graph node);
- ``local-memory``: a Triton kernel whose large per-thread array spills to local memory, captured as a graph;
- ``stack-limit``: as ``local-memory``, then ``cudaDeviceSetLimit(cudaLimitStackSize)`` lowered to 1 KiB after the
  capture, then the graph replayed and the kernel launched eagerly again (does the driver give the reservation back,
  and grow it again when a kernel needs it?).

Each case prints one JSON line: the outside-allocator bytes after every step. ``--gib`` sets the size of the
intermediates. Run inside a local GPU reservation, with or without ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

CASES = ("eager", "graph", "graph-reuse", "linear", "triton", "nodes-5000", "nodes-20000", "local-memory",
         "stack-limit")


def _cudart():
    import ctypes

    import torch

    torch.cuda.init()
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    import glob
    import os

    base = os.path.dirname(torch.__file__) + "/../nvidia"
    for path in glob.glob(base + "/*/lib/libcudart.so*"):
        return ctypes.CDLL(path)
    raise OSError("libcudart not found")


def stack_limit(value=None):
    """cudaLimitStackSize (bytes per thread): read, or set to ``value`` first."""
    import ctypes

    rt = _cudart()
    if value is not None:
        assert rt.cudaDeviceSetLimit(0, ctypes.c_size_t(value)) == 0
    out = ctypes.c_size_t()
    assert rt.cudaDeviceGetLimit(ctypes.byref(out), 0) == 0
    return out.value


def outside(torch):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return dict(outside_GiB=round((total - free - torch.cuda.memory_reserved()) / 2**30, 3),
                reserved_GiB=round(torch.cuda.memory_reserved() / 2**30, 3),
                free_GiB=round(free / 2**30, 3))


def run_case(case, gib):
    import torch

    steps = {}
    n = int(gib * 2**30 / 4 / 4)  # four float32 intermediates of n elements: about gib in all
    x = torch.randn(n, device="cuda")
    steps["start"] = outside(torch)

    def chain(v):
        a = v * 2.0
        b = a + 1.0
        c = b.sin()
        return (c * a).sum()

    if case == "eager":
        out = chain(x)
        steps["after_eager"] = outside(torch)
        del out
    elif case in ("graph", "graph-reuse"):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            chain(x)  # warm-up
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.empty_cache()
        steps["after_warmup"] = outside(torch)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = chain(x)
        steps["after_capture"] = outside(torch)
        graph.replay()
        steps["after_replay"] = outside(torch)
        if case == "graph-reuse":
            pool = graph.pool()
            second = torch.cuda.CUDAGraph()
            with torch.cuda.graph(second, pool=pool):
                out2 = chain(x * 0.5)
            steps["after_second_capture"] = outside(torch)
            second.replay()
            steps["after_second_replay"] = outside(torch)
            del second, out2
        del graph, out
        torch.cuda.empty_cache()
        steps["after_delete"] = outside(torch)
    elif case == "linear":
        m = int((gib * 2**30 / 4 / 3) ** 0.5)
        a = torch.randn(m, m, device="cuda")
        steps["after_alloc"] = outside(torch)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            (a @ a @ a).sum()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = (a @ a @ a).sum()
        steps["after_capture"] = outside(torch)
        graph.replay()
        steps["after_replay"] = outside(torch)
        del graph, out
        torch.cuda.empty_cache()
        steps["after_delete"] = outside(torch)
    elif case == "triton":
        import triton
        import triton.language as tl

        @triton.jit
        def scale(src, dst, count, BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < count
            tl.store(dst + offsets, tl.load(src + offsets, mask=mask) * 2.0, mask=mask)

        def launch(v):
            y = torch.empty_like(v)
            scale[(triton.cdiv(v.numel(), 1024),)](v, y, v.numel(), BLOCK=1024)
            return y.sum()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launch(x)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.empty_cache()
        steps["after_warmup"] = outside(torch)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = launch(x)
        steps["after_capture"] = outside(torch)
        graph.replay()
        steps["after_replay"] = outside(torch)
        del graph, out
        torch.cuda.empty_cache()
        steps["after_delete"] = outside(torch)
    elif case.startswith("nodes-"):
        count = int(case.split("-")[1])
        small = torch.zeros(1024, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            small.add_(1.0)
        torch.cuda.current_stream().wait_stream(stream)
        steps["before_capture"] = outside(torch)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(count):
                small.add_(1.0)
        steps["after_capture"] = outside(torch)
        graph.replay()
        steps["after_replay"] = outside(torch)
        del graph
        torch.cuda.empty_cache()
        steps["after_delete"] = outside(torch)
    elif case in ("local-memory", "stack-limit"):
        import triton
        import triton.language as tl

        @triton.jit
        def spill(src, dst, count, BLOCK: tl.constexpr, WIDTH: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < count
            v = tl.load(src + offsets, mask=mask)
            acc = tl.zeros([BLOCK, WIDTH], dtype=tl.float32) + v[:, None] * tl.arange(0, WIDTH)[None, :]
            acc = tl.sin(acc) * tl.cos(acc)  # a [256, 512] float32 tile per 128 threads spills to local memory
            tl.store(dst + offsets, tl.sum(acc, axis=1), mask=mask)

        def launch(v):
            y = torch.empty_like(v)
            spill[(triton.cdiv(v.numel(), 256),)](v, y, v.numel(), BLOCK=256, WIDTH=512, num_warps=4)
            return y

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launch(x)
        torch.cuda.current_stream().wait_stream(stream)
        steps["after_eager_launch"] = dict(outside(torch), stack=stack_limit())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = launch(x)
        steps["after_capture"] = outside(torch)
        graph.replay()
        steps["after_replay"] = outside(torch)
        if case == "stack-limit":
            reference = out.clone()
            steps["after_limit"] = dict(outside(torch), stack=stack_limit(1024))
            graph.replay()
            steps["after_limited_replay"] = dict(outside(torch), stack=stack_limit(),
                                                 same=bool(torch.equal(out, reference)))
            again = launch(x)
            steps["after_limited_eager"] = dict(outside(torch), stack=stack_limit(),
                                                same=bool(torch.equal(again, reference)))
        del graph, out
        torch.cuda.empty_cache()
        steps["after_delete"] = outside(torch)
    steps["end"] = outside(torch)
    print(json.dumps(dict(case=case, gib=gib, alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                          gpu=torch.cuda.get_device_name(), steps=steps)), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=CASES, help="one case in this process (default: every case, one process each)")
    p.add_argument("--gib", type=float, default=4.0)
    args = p.parse_args(argv)
    if args.case:
        run_case(args.case, args.gib)
        return
    for case in CASES:
        subprocess.run([sys.executable, __file__, "--case", case, "--gib", str(args.gib)], check=False)


if __name__ == "__main__":
    main()
