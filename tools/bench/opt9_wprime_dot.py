"""OPT9-T01 falsifier: the radial path's backward as ``dr = <dw, w'(r)>`` instead of ``da = dw W`` (plans/opt-9.md).

Today's float32 MH-1 backward of the folded radial layer (``chunked_conv_backward``) writes ``dw [E, 1280]`` in
L2-sized chunks from the streaming convolution's backward, and the Triton input gradient reads each chunk on a side
stream: ``da = dw W`` (``W [1280, 65]``, 83k FMA per edge). The radial tables' backward then forms
``dr = sum_j da_j a_j'(r)`` from the table's derivative series. With the per-piece w-tables of ideas D2
(``fastmace.kernels.triton.w_table``), the same ``dr`` is ``sum_c dw_c w_c'(r)`` with ``w'(r) = sum_t T_t(x) G'_t``
(``G' = C' W^T``, ``[terms, 1280]`` per (pair, piece)): about 10 x 1280 FMA per edge instead of 65 x 1280. Inside the
convolution backward that dot would need no ``dw`` write at all (the fp32 profile's idea, 28 October 2026).

This probe measures, on synthetic operands with MH-1's float32 shapes (degree-10 tables: 11 value terms, 10
derivative terms; edges in receiver order with random (pair, piece) groups), standalone and with the L2 flushed
before every call (flush timed alone and subtracted):

- ``chain``: what the idea removes. The Triton input gradient (the library's ``input_grad_into``) over 1,536-edge
  chunks read from two L2-sized slots, as in ``streaming_conv_folded_backward``, then the radial tables' backward
  ``dr = <da, a'(r)>`` from the derivative coefficients (a stand-in kernel with the table's shapes).
- ``dw_store``: the ``dw`` chunk writes the fused kernel would drop (a store kernel into the two slots).
- ``wdot``: what a fused kernel must add: per edge, ``w'`` from ``G'`` of its group and the dot with ``dw``, with
  ``dw`` made in registers (no ``dw`` read), edges in receiver order as the convolution owns them.
- ``wdot_sorted``: the same with the edges sorted by group (the best case for ``G'`` reuse; a receiver-owned kernel
  cannot have it).
- ``wdot_load``: ``wdot`` reading ``dw`` from a full ``[E, 1280]`` tensor (an unfused variant).

Exactness: ``wdot``'s dot against a float64 CPU reference on sampled rows (rounding-level by construction: the
series and the product are reassociated, as in D2).

Usage: python tools/bench/opt9_wprime_dot.py --cell NaNbO3 [--cell NbCl5] [--blocks 15] -> JSON lines
"""

from __future__ import annotations

import argparse
import json
import statistics
from functools import cache

import torch

# MH-1 float32, 3x3x3 (opt6-T05 certificates, interaction 1; opt8-T01 edge counts).
CELLS = {"NaNbO3": {"edges": 84072, "pairs": 9, "pieces": 65},
         "NbCl5": {"edges": 52416, "pairs": 4, "pieces": 53},
         "NaCl": {"edges": 6912, "pairs": 4, "pieces": 47}}
Q, P1, TERMS = 1280, 65, 11  # w width, [h c, c] width, value terms (degree 10); the derivative series has TERMS - 1
CHUNK = 1536  # the plan's chunk_edges (opt8-T01)
CONFIGS = ((16, 128), (32, 64), (16, 64), (32, 128))


@cache
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def wdot(x_ptr, group_ptr, dg_ptr, dw_ptr, dvec_ptr, out_ptr, E, Q, TERMS: tl.constexpr, LOAD: tl.constexpr,
             BLOCK_E: tl.constexpr, BLOCK_Q: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        rows = e < E
        x = tl.load(x_ptr + e, mask=rows, other=0.0)
        base = tl.load(group_ptr + e, mask=rows, other=0).to(tl.int64) * TERMS * Q
        scale = 1.0 + 0.001 * (e % 7).to(tl.float32)
        total = tl.zeros((BLOCK_E,), dtype=tl.float32)
        for start in range(0, Q, BLOCK_Q):
            q = start + tl.arange(0, BLOCK_Q)
            mask = rows[:, None] & (q < Q)[None, :]
            t0 = tl.full((BLOCK_E,), 1.0, tl.float32)
            t1 = x
            d = t0[:, None] * tl.load(dg_ptr + base[:, None] + q[None, :], mask=mask, other=0.0)
            d += t1[:, None] * tl.load(dg_ptr + base[:, None] + Q + q[None, :], mask=mask, other=0.0)
            for k in tl.static_range(2, TERMS):
                t0, t1 = t1, 2.0 * x * t1 - t0
                d += t1[:, None] * tl.load(dg_ptr + base[:, None] + k * Q + q[None, :], mask=mask, other=0.0)
            if LOAD:
                g = tl.load(dw_ptr + e.to(tl.int64)[:, None] * Q + q[None, :], mask=mask, other=0.0)
            else:  # dw "in registers": a cached row times a per-edge scale
                g = scale[:, None] * tl.load(dvec_ptr + q, mask=q < Q, other=0.0)[None, :]
            total += tl.sum(g * d, axis=1)
        tl.store(out_ptr + e, total, mask=rows)

    @triton.jit
    def table_backward(x_ptr, group_ptr, dcoef_ptr, da_ptr, out_ptr, E, J, TERMS: tl.constexpr,
                       BLOCK_E: tl.constexpr, BLOCK_J: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        j = tl.arange(0, BLOCK_J)
        rows = e < E
        mask = rows[:, None] & (j < J)[None, :]
        x = tl.load(x_ptr + e, mask=rows, other=0.0)
        base = tl.load(group_ptr + e, mask=rows, other=0).to(tl.int64) * TERMS * J
        t0 = tl.full((BLOCK_E,), 1.0, tl.float32)
        t1 = x
        d = t0[:, None] * tl.load(dcoef_ptr + base[:, None] + j[None, :], mask=mask, other=0.0)
        d += t1[:, None] * tl.load(dcoef_ptr + base[:, None] + J + j[None, :], mask=mask, other=0.0)
        for k in tl.static_range(2, TERMS):
            t0, t1 = t1, 2.0 * x * t1 - t0
            d += t1[:, None] * tl.load(dcoef_ptr + base[:, None] + k * J + j[None, :], mask=mask, other=0.0)
        g = tl.load(da_ptr + e.to(tl.int64)[:, None] * J + j[None, :], mask=mask, other=0.0)
        tl.store(out_ptr + e, tl.sum(g * d, axis=1), mask=rows)

    @triton.jit
    def store(slot_ptr, rows, Q, BLOCK_E: tl.constexpr, BLOCK_Q: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        mask = (e < rows)[:, None] & (q < Q)[None, :]
        tl.store(slot_ptr + e.to(tl.int64)[:, None] * Q + q[None, :], (e[:, None] + q[None, :]).to(tl.float32),
                 mask=mask)

    return wdot, table_backward, store


def operands(cell, gen, dev):
    edges, groups = cell["edges"], cell["pairs"] * cell["pieces"] + 1  # + the all-zero group, as in w_table
    td = TERMS - 1
    group = torch.randint(0, groups - 1, (edges,), generator=gen).to(torch.int32)
    x = torch.rand(edges, generator=gen) * 2 - 1
    W = torch.randn(Q, P1, generator=gen) / P1 ** 0.5
    dcoef = torch.randn(groups, td, P1, generator=gen) / td  # the table's derivative series
    gprime = torch.einsum("gtj,qj->gtq", dcoef.double(), W.double()).float().contiguous()  # G' = C' W^T
    dvec = torch.randn(Q, generator=gen)
    order = torch.argsort(group)
    ops = {"group": group, "x": x, "W": W, "dcoef": dcoef.contiguous(), "gprime": gprime, "dvec": dvec,
           "group_sorted": group[order].contiguous(), "x_sorted": x[order].contiguous()}
    ops = {k: v.to(dev) for k, v in ops.items()}
    ops["main"], ops["last"] = ops["W"][:, :P1 - 1].contiguous(), ops["W"][:, P1 - 1].contiguous()
    ops["slots"] = torch.randn(2, CHUNK, Q, generator=gen).to(dev)
    ops["da"] = torch.empty(edges, P1, device=dev)
    ops["dr"] = torch.empty(edges, device=dev)
    ops["dw"] = torch.randn(edges, Q, generator=gen).to(dev)
    return ops


def variants(ops, edges, config):
    import triton

    from fastmace.kernels.triton.skinny_linear import input_grad_into

    wdot, table_backward, store = _kernels()
    be, bq = config
    td = TERMS - 1

    def chain():
        for chunk, start in enumerate(range(0, edges, CHUNK)):
            stop = min(edges, start + CHUNK)
            input_grad_into(ops["slots"][chunk % 2, :stop - start], ops["W"], ops["main"], ops["last"],
                            ops["da"][start:stop])
        table_backward[(triton.cdiv(edges, 32),)](ops["x"], ops["group"], ops["dcoef"], ops["da"], ops["dr"], edges,
                                                  P1, TERMS=td, BLOCK_E=32, BLOCK_J=128)

    def dw_store():
        for chunk, start in enumerate(range(0, edges, CHUNK)):
            rows = min(edges, start + CHUNK) - start
            store[(triton.cdiv(rows, 32), triton.cdiv(Q, 128))](ops["slots"][chunk % 2], rows, Q, BLOCK_E=32,
                                                                 BLOCK_Q=128)

    def dot(x, group, load):
        return lambda: wdot[(triton.cdiv(edges, be),)](x, group, ops["gprime"], ops["dw"], ops["dvec"], ops["dr"],
                                                        edges, Q, TERMS=td, LOAD=load, BLOCK_E=be, BLOCK_Q=bq)

    return {"chain": chain, "dw_store": dw_store, "wdot": dot(ops["x"], ops["group"], False),
            "wdot_sorted": dot(ops["x_sorted"], ops["group_sorted"], False),
            "wdot_load": dot(ops["x"], ops["group"], True)}


def graph_of(fn, flush, calls):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
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


def exactness(ops, edges, config):
    """``wdot`` with ``dw`` loaded, against float64 on 256 sampled rows (relative to the row's absolute dot)."""
    fns = variants(ops, edges, config)
    fns["wdot_load"]()
    torch.cuda.synchronize()
    gen = torch.Generator().manual_seed(29)
    rows = torch.randperm(edges, generator=gen)[:256]
    x = ops["x"].cpu().double()[rows]
    cheb = [torch.ones_like(x), x]
    for _ in range(2, TERMS - 1):
        cheb.append(2 * x * cheb[-1] - cheb[-2])
    basis = torch.stack(cheb, dim=1)
    gp = ops["gprime"].cpu().double()[ops["group"].cpu()[rows].long()]
    wp = torch.einsum("et,etq->eq", basis, gp)
    dw = ops["dw"].cpu().double()[rows]
    ref = (dw * wp).sum(1)
    got = ops["dr"].cpu().double()[rows]
    return float(((got - ref).abs() / (dw * wp).abs().sum(1)).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", action="append", choices=sorted(CELLS), help="default NaNbO3 and NbCl5")
    p.add_argument("--blocks", type=int, default=15, help="alternating timing blocks")
    p.add_argument("--calls", type=int, default=20, help="calls per CUDA graph")
    p.add_argument("--label", default="")
    args = p.parse_args()
    dev = torch.device("cuda")
    l2 = torch.cuda.get_device_properties(dev).L2_cache_size
    flush_buf = torch.empty(2 * l2 // 4 + 1024, device=dev)

    def flush():
        flush_buf.fill_(1.0)

    for name in args.cell or ["NaNbO3", "NbCl5"]:
        cell = CELLS[name]
        gen = torch.Generator().manual_seed(20261028)
        ops = operands(cell, gen, dev)
        edges = cell["edges"]
        record = {"cell": name, "label": args.label, "gpu": torch.cuda.get_device_name(dev),
                  "capability": "sm_%d%d" % torch.cuda.get_device_capability(dev), "l2_MiB": l2 / 2**20,
                  "edges": edges, "groups": cell["pairs"] * cell["pieces"] + 1,
                  "gprime_MiB": ops["gprime"].numel() * 4 / 2**20, "torch": torch.__version__}
        # Pick wdot's tile from the configs on one quick pass (receiver order), then time everything with it.
        quick = {}
        for config in CONFIGS:
            g = graph_of(variants(ops, edges, config)["wdot"], flush, args.calls)
            quick[config] = min(replay_ms(g, args.calls) for _ in range(3))
        config = min(quick, key=quick.get)
        record["wdot_config"] = "%dx%d" % config
        fns = variants(ops, edges, config)
        fns["flush"] = lambda: None
        graphs = {k: graph_of(fn, flush, args.calls) for k, fn in fns.items()}
        times = {k: [] for k in graphs}
        for _ in range(args.blocks):
            for k, g in graphs.items():
                times[k].append(replay_ms(g, args.calls))
        flush_ms = statistics.median(times.pop("flush"))
        record["flush_ms"] = flush_ms
        for k, v in times.items():
            record[f"{k}_ms"] = statistics.median(v) - flush_ms
            record[f"{k}_spread"] = (max(v) - min(v)) / statistics.median(v)
        record["removed_ms"] = record["chain_ms"] + record["dw_store_ms"]
        record["net_ms"] = record["removed_ms"] - record["wdot_ms"]
        record["wdot_max_rel_err"] = exactness(ops, edges, config)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
