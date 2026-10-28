"""OPT10-T01 falsifier: the gate's backward folded into ``linear_2``'s input-gradient GEMM (plans/opt-10.md).

In MH-1's gated chain (``fastmace.rewrites.fusion.GatedRows`` with ``segment_gemm``), the backward of
``linear_2`` writes, per gated segment l (2l + 1 = D rows per atom), ``drows = grad_l @ W_l^T``
(``fastmace::joined_products_input_grads``, ``[atoms * D, 512]``). A compiled elementwise kernel then reads
``drows``, the gated rows and the atom's gates and writes ``dgated = drows * gate`` and
``dgate = sum_m drows * gated`` (``[atoms, 512]``).

The fused kernel tiles atoms instead of rows: per (atom block, column block) it runs one K-loop per component
m, then multiplies by the gate tile (loaded once) and adds ``acc * gated_m`` into ``dgate`` in registers. No
``drows`` is written or read again, no atomics, and each ``dgated`` element is the same product as the chain's
(the K-loop is the segment GEMM's; only ``dgate``'s m-sum may be ordered differently).

Synthetic operands with MH-1's shapes (``linear_2``: 512 channels per segment, segments 0e, 1o, 2e, 3o):
NbCl5 1,944 atoms, NaNbO3 1,080, NaCl 216. Both sides run the 0e segment's plain product too. Timing as in
OPT9-T01: CUDA graphs of 20 calls, L2 flushed before each call (flush subtracted), 15 alternating blocks.

Usage: python tools/bench/opt10_gate_epilogue.py [--cell NbCl5 ...] -> JSON lines
"""

from __future__ import annotations

import argparse
import json
import statistics
from functools import cache

import torch

CELLS = {"NbCl5": 1944, "NaNbO3": 1080, "NaCl": 216}
MUL, DIMS = 512, (1, 3, 5, 7)


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fused(g_ptr, wt_ptr, gated_ptr, gate_ptr, dgated_ptr, dgate_ptr, atoms_n, N, K, width, mul_out,
              gate_stride, D: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pa, pn = tl.program_id(0), tl.program_id(1)
        atom = (pa * BLOCK_A + tl.arange(0, BLOCK_A)).to(tl.int64)
        j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        amask, jmask = atom < atoms_n, j < N
        tile = amask[:, None] & jmask[None, :]
        gate = tl.load(gate_ptr + atom[:, None] * gate_stride + j[None, :], mask=tile, other=0.0)
        dg = tl.zeros((BLOCK_A, BLOCK_N), dtype=tl.float32)
        kk = tl.arange(0, BLOCK_K)
        for m in tl.static_range(D):
            acc = tl.zeros((BLOCK_A, BLOCK_N), dtype=tl.float32)
            for start in range(0, K, BLOCK_K):
                k = start + kk
                a = tl.load(g_ptr + (atom * width + m * mul_out)[:, None] + k[None, :],
                            mask=amask[:, None] & (k < K)[None, :], other=0.0)
                b = tl.load(wt_ptr + k[:, None] * N + j[None, :], mask=(k < K)[:, None] & jmask[None, :], other=0.0)
                acc += tl.dot(a, b, input_precision="ieee")
            row = atom * D + m
            gm = tl.load(gated_ptr + row[:, None] * N + j[None, :], mask=tile, other=0.0)
            tl.store(dgated_ptr + row[:, None] * N + j[None, :], acc * gate, mask=tile)
            dg += acc * gm
        tl.store(dgate_ptr + atom[:, None] * N + j[None, :], dg, mask=tile)

    return fused


def operands(atoms, gen, dev):
    width = MUL * sum(DIMS)
    offsets, at = [], 0
    for d in DIMS:
        offsets.append(at)
        at += d * MUL
    return {
        "grad": torch.randn(atoms, width, generator=gen).to(dev),
        "wt": [(torch.randn(MUL, MUL, generator=gen) / MUL ** 0.5).to(dev).contiguous() for _ in DIMS],
        "gated": [torch.randn(atoms * d, MUL, generator=gen).to(dev) for d in DIMS[1:]],
        "gates": torch.rand(atoms, MUL * (len(DIMS) - 1), generator=gen).to(dev),
        "offsets": offsets, "width": width, "atoms": atoms,
    }


def gate_backward(drows, gated, gates, atoms):
    """The chain's elementwise part (compiled): per gated segment ``dgated`` and ``dgate``."""
    out = []
    for i, (dr, gd) in enumerate(zip(drows, gated)):
        d = DIMS[i + 1]
        gate = gates[:, i * MUL:(i + 1) * MUL]
        dr3 = dr.view(atoms, d, MUL)
        out.append(((dr3 * gate.unsqueeze(1)).reshape(atoms * d, MUL), (dr3 * gd.view(atoms, d, MUL)).sum(1)))
    return out


def chain_fn(ops, compiled):
    from fastmace.kernels.triton.segment_gemm import joined_products_input_grads

    def run():
        drows = joined_products_input_grads(ops["grad"], ops["wt"], ops["offsets"], list(DIMS), "ieee")
        return drows[0], compiled(drows[1:], ops["gated"], ops["gates"], ops["atoms"])
    return run


def fused_fn(ops, config):
    import triton

    from fastmace.kernels.triton.segment_gemm import joined_products_input_grads

    block_a, block_n, block_k, warps, stages = config
    atoms, width = ops["atoms"], ops["width"]
    outs = [(torch.empty(atoms * d, MUL, device=ops["grad"].device), torch.empty(atoms, MUL, device=ops["grad"].device))
            for d in DIMS[1:]]
    kernel = _kernel()

    def run():
        # The 0e segment has no gate: its product is the chain's (same library call, one product).
        d0 = joined_products_input_grads(ops["grad"], ops["wt"][:1], [0], [1], "ieee")[0]
        for i, d in enumerate(DIMS[1:]):
            dgated, dgate = outs[i]
            g = ops["grad"].reshape(-1)[ops["offsets"][i + 1]:]
            gates = ops["gates"].reshape(-1)[i * MUL:]
            kernel[(triton.cdiv(atoms, block_a), triton.cdiv(MUL, block_n))](
                g, ops["wt"][i + 1], ops["gated"][i], gates, dgated, dgate, atoms, MUL, MUL, width, MUL,
                ops["gates"].stride(0), D=d, BLOCK_A=block_a, BLOCK_N=block_n, BLOCK_K=block_k,
                num_warps=warps, num_stages=stages)
        return d0, outs
    return run


def graph_of(fn, flush, calls):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            flush()
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            flush()
            fn()
    return graph


def replay_ms(graph, calls):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / calls


CONFIGS = ((64, 64, 32, 4, 3), (32, 64, 32, 4, 3), (64, 32, 32, 4, 3), (32, 128, 32, 4, 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", action="append", choices=sorted(CELLS))
    p.add_argument("--blocks", type=int, default=15)
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--label", default="")
    args = p.parse_args()
    dev = torch.device("cuda")
    l2 = torch.cuda.get_device_properties(dev).L2_cache_size
    flush_buf = torch.empty(2 * l2 // 4 + 1024, device=dev)
    compiled = torch.compile(gate_backward, dynamic=False)

    def flush():
        flush_buf.fill_(1.0)

    for name in args.cell or ["NbCl5", "NaNbO3"]:
        atoms = CELLS[name]
        ops = operands(atoms, torch.Generator().manual_seed(20261029), dev)
        chain = chain_fn(ops, compiled)
        record = {"cell": name, "atoms": atoms, "label": args.label, "gpu": torch.cuda.get_device_name(dev),
                  "capability": "sm_%d%d" % torch.cuda.get_device_capability(dev), "torch": torch.__version__}
        # Exactness of each config against the chain, then pick the fastest config on a quick pass.
        _, ref = chain()
        quick, errors = {}, {}
        for config in CONFIGS:
            fn = fused_fn(ops, config)
            _, got = fn()
            torch.cuda.synchronize()
            err_gated = max(float((g[0] - r[0]).abs().max() / r[0].abs().max()) for g, r in zip(got, ref))
            err_gate = max(float((g[1] - r[1]).abs().max() / r[1].abs().max()) for g, r in zip(got, ref))
            bitwise = all(torch.equal(g[0], r[0]) for g, r in zip(got, ref))
            errors["%dx%dx%d" % config[:3]] = {"dgated_rel": err_gated, "dgate_rel": err_gate,
                                               "dgated_bitwise": bitwise}
            g = graph_of(fn, flush, args.calls)
            quick[config] = min(replay_ms(g, args.calls) for _ in range(3))
        config = min(quick, key=quick.get)
        record.update(config="%dx%dx%d" % config[:3], errors=errors,
                      quick_ms={"%dx%dx%d" % c[:3]: v for c, v in quick.items()})
        # Keep every closure alive while its graph replays: the fused one owns its output buffers, and with
        # expandable segments freed memory can be unmapped (an illegal address in the replay).
        fns = {"chain": chain, "fused": fused_fn(ops, config), "flush": lambda: None}
        graphs = {k: graph_of(fn, flush, args.calls) for k, fn in fns.items()}
        times = {k: [] for k in graphs}
        for _ in range(args.blocks):
            for k, g in graphs.items():
                try:
                    times[k].append(replay_ms(g, args.calls))
                except Exception as error:
                    raise RuntimeError(f"replay of {k!r} failed") from error
        flush_ms = statistics.median(times.pop("flush"))
        for k, v in times.items():
            record[f"{k}_ms"] = statistics.median(v) - flush_ms
            record[f"{k}_spread"] = (max(v) - min(v)) / statistics.median(v)
        record["saved_ms"] = record["chain_ms"] - record["fused_ms"]
        record["fused_over_chain"] = record["fused_ms"] / record["chain_ms"]
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
