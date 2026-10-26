"""OPT6-T01 falsifier: L2-resident edge chunks for the folded radial last layer and the streaming convolution.

The second MH-1 interaction materialises its per-edge weights ``w = a Wᵀ`` (``[E, 65] @ [65, 1280]``) in DRAM,
the receiver-streaming forward reads them back, the edge-owned backward reads them again and writes
``dw [E, 1280]``, and the folded layer's Triton input gradient reads ``dw``. On NaNbO3 each of those tensors is
about 423 MB. The 14 October screen (ResearchModule ``studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md``, step 4.1)
chunked the edges with the unpatched OEq, which returns a full-size output per call, and lost 1.4-2x. The
streaming kernels are vendored here and address edges through receiver-sorted CSR rows, so a contiguous
receiver range is a contiguous edge range, the forward writes only its own rows and the edge-owned backward
already adds sender gradients atomically into one buffer. This probe measures whether chunks sized to the L2
then win:

- ``base``: production order: one GEMM into ``w``, streaming forward; streaming backward into ``dw``, Triton
  input gradient (the ``folded_linear`` kernel with the ``waves`` tile rule, as the plan's default).
- ``store``: per receiver-aligned chunk, the GEMM writes its slice of the stored ``w`` and the range forward
  reads it at once (from L2); in the backward, per chunk, the streaming backward writes a chunk-sized ``dw``
  scratch that the input gradient reads at once. ``w`` is still stored for the backward.
- ``recompute``: as ``store``, but ``w`` is never stored: the forward and the backward both build each chunk's
  ``w`` into a scratch buffer (one more GEMM, no ``[E, 1280]`` tensor at all).
- ``pipe``: as ``store``, but the forward's GEMMs run on a side stream at most two chunks ahead of the
  convolution, and the backward's input gradients run on a side stream over two ``dw`` slots.
- ``mixed``: the unchunked forward, then ``pipe``'s backward (fixed edge ranges would do for it: the
  edge-owned backward needs no receiver alignment).
- ``combo``: ``pipe``'s forward on ``--forward-chunk-edges`` chunks, ``pipe``'s backward on ``--chunk-edges``.

Operands: the real ``x``, ``sh``, senders, receivers, CSR rows and output gradient of one force call of the
float32 hybrid plan (recorded at the streaming kernels' Python launch), with a random ``a`` and ``W`` of the
production shapes (timing does not depend on their values; the chunked results are checked against ``base``
on the same values). Padded edges beyond the compacted count get no weight in the chunked variants (their
cutoff and its derivative are zero, so their input gradient is zero in production too).

Each graph starts by overwriting a buffer of twice the L2 size, so no variant inherits a warm L2 from the
previous replay; the flush alone is timed and subtracted. Variants alternate block by block in one process.

Stop rule (plans/opt-6.md): forward + backward must gain at least 1.0 ms on NaNbO3 and 0.6 ms on NbCl5.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5", "NaNbO3"])
    p.add_argument("--chunk-edges", type=int, nargs="+", default=[2048, 4096, 6144, 8192, 12288])
    p.add_argument("--variants", nargs="+", default=["store", "recompute", "pipe", "mixed"],
                   choices=["store", "recompute", "pipe", "mixed", "combo"])
    p.add_argument("--forward-chunk-edges", type=int, default=0,
                   help="combo: pipelined forward chunks (0: unchunked forward); backward chunks: --chunk-edges")
    p.add_argument("--reps", type=int, default=20, help="Replays of each variant per captured graph")
    p.add_argument("--blocks", type=int, default=9, help="Alternating timing blocks")
    p.add_argument("--rattle-A", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261026)
    return p.parse_args(argv)


def range_forward_source(source):
    """The streaming forward as ``oeq_projected_forward_range``: threads own receivers node_begin..node_stop-1
    and the edge range of each comes from the global CSR rows, so senders and edges keep global indices."""
    start = source.index('extern "C" __global__ void oeq_projected_forward(')
    end = source.index('extern "C"', start + 10)
    body = source[start:end]
    edits = [
        (r"oeq_projected_forward\(", "oeq_projected_forward_range("),
        (r"int64_t edge_count,", "int64_t edge_count,\n    int64_t node_begin,\n    int64_t node_stop,"),
        (r"node_channel_count\s*=\s*node_count\s*\*\s*CHANNEL_DIM;",
         "node_channel_count = (node_stop - node_begin) * CHANNEL_DIM;"),
        (r"const int64_t node\s*=\s*node_channel_index\s*/\s*CHANNEL_DIM;",
         "const int64_t local_node = node_channel_index / CHANNEL_DIM;\n"
         "    const int64_t node = node_begin + local_node;"),
        (r"int\(node_channel_index\s*-\s*node\s*\*\s*CHANNEL_DIM\)",
         "int(node_channel_index - local_node * CHANNEL_DIM)"),
    ]
    for pattern, replacement in edits:
        body, count = re.subn(pattern, replacement, body, count=1)
        if count != 1:
            raise RuntimeError(f"forward template changed: {pattern!r} not found")
    return source + "\n" + body


class Recorder:
    """Copies of the streaming kernels' operands from one force call."""

    def __init__(self, torch):
        self.torch = torch
        self.active = False
        self.forward = None
        self.backward = None
        self.kernels = None

    def install(self):
        from fastmace.kernels.oeq_streaming.runtime import StreamingKernels

        forward, backward = StreamingKernels.forward, StreamingKernels.backward
        recorder = self

        def record_forward(kernels, x, sh, weights, senders, row_ptr, out):
            forward(kernels, x, sh, weights, senders, row_ptr, out)
            if recorder.active:
                recorder.kernels = kernels
                recorder.forward = dict(x=x.clone(), sh=sh.clone(), weights_shape=tuple(weights.shape),
                                        senders=senders.clone(), row_ptr=row_ptr.clone())

        def record_backward(kernels, x, sh, weights, senders, receivers, dout, dx, dsh, dweights):
            backward(kernels, x, sh, weights, senders, receivers, dout, dx, dsh, dweights)
            if recorder.active:
                recorder.backward = dict(receivers=receivers.clone(), dout=dout.clone())

        StreamingKernels.forward, StreamingKernels.backward = record_forward, record_backward


def receiver_chunks(row_ptr, target):
    """(r0, r1, e0, e1) receiver-aligned chunks of about ``target`` real edges each."""
    import numpy as np

    rows = row_ptr.cpu().numpy().astype(np.int64)
    nodes, real = rows.shape[0] - 1, int(rows[-1])
    chunks, r0 = [], 0
    while r0 < nodes:
        e0 = int(rows[r0])
        r1 = int(np.searchsorted(rows, e0 + target, side="right")) - 1
        r1 = min(max(r1, r0 + 1), nodes)
        chunks.append((r0, r1, e0, int(rows[r1])))
        r0 = r1
    assert chunks[-1][3] == real
    return chunks


class Probe:
    def __init__(self, torch, kernels, operands, seed):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming import runtime
        from fastmace.kernels.triton import skinny_linear, tile_rule

        self.torch, self.k = torch, kernels
        f, b = operands.forward, operands.backward
        self.x, self.sh, self.senders, self.row_ptr = f["x"], f["sh"], f["senders"], f["row_ptr"]
        self.receivers, self.dout = b["receivers"], b["dout"]
        self.nodes, self.edges = self.x.shape[0], self.sh.shape[0]
        self.real = int(self.row_ptr[-1])
        self.q = kernels.weight_dim
        assert f["weights_shape"] == (self.edges, self.q), f["weights_shape"]
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.a = torch.randn((self.edges, 65), device="cuda", generator=g)
        self.a[self.real:] = 0.0  # padding: [h c, c] with c = 0, so w and the edge's message are zero
        weight = torch.randn((self.q, 65), device="cuda", generator=g) * 0.1
        self.wt = weight.t().contiguous()  # [65, Q]: the same GEMM in every variant
        self.main, self.last = weight[:, :64].contiguous(), weight[:, 64].contiguous()
        self.sms = tile_rule.device_sms(self.a.device)
        self.skinny = skinny_linear
        self.w = torch.empty((self.edges, self.q), device="cuda")
        self.dw = torch.empty((self.edges, self.q), device="cuda")
        self.out = torch.empty((self.nodes, kernels.output_dim), device="cuda")
        self.dx = torch.zeros_like(self.x)
        self.dsh = torch.empty_like(self.sh)
        self.da = torch.zeros_like(self.a)
        cubin = runtime.compile_cubin(range_forward_source(kernels.source), torch.cuda.get_device_capability())
        self._module = runtime._check(driver.cuModuleLoadData(cubin))
        self.range_fn = runtime._check(driver.cuModuleGetFunction(self._module, b"oeq_projected_forward_range"))
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        self.l2_bytes = int(l2)
        self.flush_buffer = torch.empty(2 * self.l2_bytes // 4, device="cuda")
        self.scratch_rows = 0
        self.side = torch.cuda.Stream()

    # -- pieces -------------------------------------------------------------------------------------
    def input_grad(self, grad, out):
        import triton

        rows = grad.shape[0]
        block_m, block_k, warps, stages = self.skinny.tile_choice(rows, self.sms, "waves")
        self.skinny._triton_kernel()[(triton.cdiv(rows, block_m),)](
            grad, self.main, self.last, out, rows, self.q, 64, BLOCK_M=block_m, BLOCK_P=64, BLOCK_K=block_k,
            num_warps=warps, num_stages=stages)

    def range_forward(self, r0, r1, weights_ptr):
        k = self.k
        blocks = ((r1 - r0) * k.channels + k.num_threads - 1) // k.num_threads
        p, i = ctypes.c_void_p, ctypes.c_longlong
        k._launch(self.range_fn, blocks,
                  (self.nodes, self.edges, r0, r1, self.x.data_ptr(), self.sh.data_ptr(), weights_ptr,
                   self.senders.data_ptr(), self.row_ptr.data_ptr(), self.out.data_ptr()),
                  (i, i, i, i, p, p, p, p, p, p))

    def edge_backward(self, e0, e1, weights_ptr, dweights_ptr):
        k, f = self.k, 4  # float32 / int32 bytes
        n = e1 - e0
        blocks = (n * k.cohort + k.num_threads - 1) // k.num_threads
        p, i = ctypes.c_void_p, ctypes.c_longlong
        k._launch(k._backward, blocks,
                  (self.nodes, n, self.x.data_ptr(), self.sh.data_ptr() + e0 * k.edge_dim * f, weights_ptr,
                   self.senders.data_ptr() + e0 * f, self.receivers.data_ptr() + e0 * f, self.dout.data_ptr(),
                   self.dx.data_ptr(), self.dsh.data_ptr() + e0 * k.edge_dim * f, dweights_ptr),
                  (i, i, p, p, p, p, p, p, p, p, p))

    # -- variants -----------------------------------------------------------------------------------
    def flush(self):
        self.flush_buffer.fill_(1.0)

    def base_forward(self):
        self.torch.mm(self.a, self.wt, out=self.w)
        self.k.forward(self.x, self.sh, self.w, self.senders, self.row_ptr, self.out)

    def base_backward(self):
        self.dx.zero_()
        self.k.backward(self.x, self.sh, self.w, self.senders, self.receivers, self.dout, self.dx, self.dsh,
                        self.dw)
        self.input_grad(self.dw, self.da)

    def allocate_scratch(self, rows):
        """Chunk scratch for the largest chunk, once: captured graphs keep these addresses. ``dw_c`` has two
        slots, so the pipelined backward can write chunk k + 1 while the input gradient reads chunk k."""
        self.w_c = self.torch.empty((rows, self.q), device="cuda")
        self.dw_c = self.torch.empty((2 * rows, self.q), device="cuda")
        self.scratch_rows = rows

    def set_chunks(self, chunks):
        if max(e1 - e0 for _, _, e0, e1 in chunks) > self.scratch_rows:
            raise ValueError("chunk larger than the scratch")
        self.chunks = chunks

    def store_forward(self):
        for r0, r1, e0, e1 in self.chunks:
            self.torch.mm(self.a[e0:e1], self.wt, out=self.w[e0:e1])
            self.range_forward(r0, r1, self.w.data_ptr())

    def store_backward(self):
        self.dx.zero_()
        self.da[self.real:].zero_()
        row = self.q * 4
        for _, _, e0, e1 in self.chunks:
            self.edge_backward(e0, e1, self.w.data_ptr() + e0 * row, self.dw_c.data_ptr())
            self.input_grad(self.dw_c[:e1 - e0], self.da[e0:e1])

    def recompute_forward(self):
        row = self.q * 4
        for r0, r1, e0, e1 in self.chunks:
            self.torch.mm(self.a[e0:e1], self.wt, out=self.w_c[:e1 - e0])
            self.range_forward(r0, r1, self.w_c.data_ptr() - e0 * row)  # global edge e -> w_c[e - e0]

    def recompute_backward(self):
        self.dx.zero_()
        self.da[self.real:].zero_()
        for _, _, e0, e1 in self.chunks:
            self.torch.mm(self.a[e0:e1], self.wt, out=self.w_c[:e1 - e0])
            self.edge_backward(e0, e1, self.w_c.data_ptr(), self.dw_c.data_ptr())
            self.input_grad(self.dw_c[:e1 - e0], self.da[e0:e1])

    def pipe_forward(self, chunks=None):
        """``store_forward`` with the GEMMs on a side stream, at most two chunks ahead of the convolution: the
        GEMM fills the SMs that a chunk's few receivers leave idle. ``chunks`` defaults to ``self.chunks``."""
        torch = self.torch
        main, side = torch.cuda.current_stream(), self.side
        side.wait_stream(main)
        conv_done = []
        for k, (r0, r1, e0, e1) in enumerate(self.chunks if chunks is None else chunks):
            with torch.cuda.stream(side):
                if k >= 2:
                    side.wait_event(conv_done[k - 2])
                torch.mm(self.a[e0:e1], self.wt, out=self.w[e0:e1])
                gemm_done = torch.cuda.Event()
                gemm_done.record(side)
            main.wait_event(gemm_done)
            self.range_forward(r0, r1, self.w.data_ptr())
            event = torch.cuda.Event()
            event.record(main)
            conv_done.append(event)
        main.wait_stream(side)

    def pipe_backward(self):
        """``store_backward`` with the input gradients on a side stream, two ``dw`` slots."""
        torch = self.torch
        main, side = torch.cuda.current_stream(), self.side
        self.dx.zero_()
        self.da[self.real:].zero_()
        side.wait_stream(main)
        row = self.q * 4
        grad_done = []
        for k, (_, _, e0, e1) in enumerate(self.chunks):
            slot = (k % 2) * self.scratch_rows
            if k >= 2:
                main.wait_event(grad_done[k - 2])
            self.edge_backward(e0, e1, self.w.data_ptr() + e0 * row, self.dw_c[slot].data_ptr())
            backward_done = torch.cuda.Event()
            backward_done.record(main)
            with torch.cuda.stream(side):
                side.wait_event(backward_done)
                self.input_grad(self.dw_c[slot:slot + e1 - e0], self.da[e0:e1])
                event = torch.cuda.Event()
                event.record(side)
            grad_done.append(event)
        main.wait_stream(side)

    def results(self):
        t = self.torch
        t.cuda.synchronize()
        return dict(out=self.out.clone(), dx=self.dx.clone(), dsh=self.dsh[:self.real].clone(),
                    da=self.da[:self.real].clone())


def rel(got, ref):
    scale = float(ref.abs().max())
    return float((got - ref).abs().max()) / scale if scale else float((got - ref).abs().max())


def capture(torch, fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    return graph


def alternate(torch, graphs, reps, blocks):
    """Median ms per call of each captured graph, the graphs replayed in turn, block by block."""
    times = {name: [] for name in graphs}
    for block in range(blocks):
        order = list(graphs) if block % 2 == 0 else list(reversed(graphs))
        for name in order:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            graphs[name].replay()
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - t0) * 1e3 / reps)
    return {name: dict(median_ms=statistics.median(v), blocks_ms=[round(x, 4) for x in v])
            for name, v in times.items()}


def run_structure(torch, fm, model, args, name, atoms, recorder):
    ff = fm.ForceField(model, atoms, precision="float32")
    plan = ff.evaluation_plan(atoms)
    decision = list(plan.decision("RewriteConfig.streaming_conv"))
    ff.evaluate(atoms)  # warm-up: compiles
    recorder.active = True
    ff.evaluate(atoms)
    torch.cuda.synchronize()
    recorder.active = False
    if recorder.forward is None or recorder.backward is None:
        raise RuntimeError(f"{name}: the streaming kernels did not run (decision {decision})")
    probe = Probe(torch, recorder.kernels, recorder, args.seed)
    del ff
    torch.cuda.empty_cache()

    row = dict(natoms=probe.nodes, edges=probe.edges, real_edges=probe.real, weight_dim=probe.q,
               w_MB=probe.edges * probe.q * 4 / 1e6, l2_MB=probe.l2_bytes / 1e6, streaming_decision=decision,
               chunks={})

    def with_flush(fn):
        def step():
            probe.flush()
            fn()
        return step

    def both(forward, backward):
        def step():
            forward()
            backward()
        return step

    probe.base_forward()
    probe.base_backward()
    reference = probe.results()
    graphs = {"flush": capture(torch, probe.flush, args.reps),
              "base_fwd": capture(torch, with_flush(probe.base_forward), args.reps),
              "base_bwd": capture(torch, with_flush(probe.base_backward), args.reps),
              "base": capture(torch, with_flush(both(probe.base_forward, probe.base_backward)), args.reps)}
    chunk_lists = {size: receiver_chunks(probe.row_ptr, size) for size in args.chunk_edges}
    probe.allocate_scratch(max(e1 - e0 for chunks in chunk_lists.values() for *_, e0, e1 in chunks))
    if args.forward_chunk_edges:
        forward_chunks = receiver_chunks(probe.row_ptr, args.forward_chunk_edges)
        combo_forward = lambda: probe.pipe_forward(forward_chunks)  # noqa: E731
        row["forward_chunk_edges"] = args.forward_chunk_edges
    else:
        combo_forward = probe.base_forward
    for size, chunks in chunk_lists.items():
        probe.set_chunks(chunks)
        checks = {}
        variants = dict(store=(probe.store_forward, probe.store_backward),
                        recompute=(probe.recompute_forward, probe.recompute_backward),
                        pipe=(probe.pipe_forward, probe.pipe_backward),
                        mixed=(probe.base_forward, probe.pipe_backward),
                        combo=(combo_forward, probe.pipe_backward))
        for variant in args.variants:
            forward, backward = variants[variant]
            forward()
            backward()
            got = probe.results()
            checks[variant] = {key: rel(got[key], reference[key]) for key in reference}
            graphs[f"{variant}{size}_fwd"] = capture(torch, with_flush(forward), args.reps)
            graphs[f"{variant}{size}_bwd"] = capture(torch, with_flush(backward), args.reps)
            graphs[f"{variant}{size}"] = capture(torch, with_flush(both(forward, backward)), args.reps)
        row["chunks"][size] = dict(count=len(chunks), max_edges=max(e1 - e0 for *_, e0, e1 in chunks),
                                   chunk_MB=max(e1 - e0 for *_, e0, e1 in chunks) * probe.q * 4 / 1e6,
                                   max_rel_diff=checks)
    timing = alternate(torch, graphs, args.reps, args.blocks)
    flush = timing.pop("flush")["median_ms"]
    row["flush_ms"] = flush
    row["timing_ms"] = {key: round(value["median_ms"] - flush, 4) for key, value in timing.items()}
    row["timing_raw"] = timing
    base = row["timing_ms"]["base"]
    row["gain_ms"] = {key: round(base - value, 4) for key, value in row["timing_ms"].items()
                      if not key.endswith(("_fwd", "_bwd")) and key != "base"}
    return row


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch

    import fastmace as fm

    sys.path.insert(0, str(TOOLS))
    import ab_step_time  # noqa: E402  (tools/: commit, structures)

    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    recorder = Recorder(torch)
    recorder.install()
    model = fm.load(args.model)
    import triton

    summary = dict(tool="opt6_l2_chunked_conv", created_utc=datetime.now(timezone.utc).isoformat(),
                   host=os.uname().nodename, fastmace_commit=ab_step_time.fastmace_commit(), model=args.model,
                   gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                   torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
                   python=sys.executable, reps=args.reps, blocks=args.blocks, rattle_A=args.rattle_A,
                   seed=args.seed, structures={})
    for spec in args.structures:
        name, atoms = ab_step_time.read_structure(spec, None)
        rng = np.random.default_rng(args.seed)
        atoms.positions += rng.uniform(-args.rattle_A, args.rattle_A, atoms.positions.shape)
        stem = Path(str(name)).stem
        row = run_structure(torch, fm, model, args, stem, atoms, recorder)
        summary["structures"][stem] = row
        (args.output / f"{stem}.json").write_text(json.dumps(row, indent=2) + "\n")
        print(stem, json.dumps(dict(edges=row["edges"], timing_ms=row["timing_ms"], gain_ms=row["gain_ms"])),
              flush=True)
        torch.cuda.empty_cache()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    main()
