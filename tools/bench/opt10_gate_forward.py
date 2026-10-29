"""OPT10 forward-half falsifier: the merged residual products with the gate in their epilogue (plans/opt-10.md).

In MH-1's folded forward the gated chain starts with ``x = message / (density * beta + alpha)`` and
``first.output_rows(x, node_feats)`` (``ResidualBlockLinear``): per output segment l, the rows of x's and
node_feats' segments of that irrep are concatenated along K (a copy) and multiplied by the merged weight (cuBLAS).
Then a compiled elementwise kernel reads the outputs again and writes the scalars ``silu(s) * c_s``, the gates
``sigmoid(g) * c_g`` and per gated segment ``v * gate`` (``GatedRows.forward``).

The fused side runs one Triton kernel per output segment that reads x's and node_feats' rows in place (no
concatenation), scales the message K range's sum by the atom's ``1 / (density * beta + alpha)`` before adding the
node_feats K range (rounding: ``(x W) / r`` instead of ``(x / r) W``), and applies the gate in its epilogue:
- 0e: stores ``silu(acc) * c_s`` for the scalar columns and ``sigmoid(acc) * c_g`` for the gate columns;
- each gated segment (D = 2l + 1): tiles atoms, runs one K-loop per component m, loads the atom's gate tile once
  and stores ``acc * gate`` (the gate is ready: the 0e kernel runs first).
The backward would keep ``X = v * gate`` and ``sigma``: ``sigma'(g) v = (1 - sigma) X``, so v need not be stored.

Synthetic operands with interaction 1's shapes (MH-1: linear_1 ``128x0e+128x0e+3x128x1o+3x128x2e+2x128x3o``,
linear_res ``128x0e+128x1o``, outputs ``2048x0e+512x1o+512x2e+512x3o``). Forward only: what each side saves for
the backward is not counted. Timing as OPT10-T01: CUDA graphs of 20 calls, L2 flushed per call (subtracted),
alternating blocks, medians.

Usage: python tools/bench/opt10_gate_forward.py [--cell NbCl5 ...] -> JSON lines
"""

from __future__ import annotations

import argparse
import json
import statistics
from functools import cache

import torch

CELLS = {"NbCl5": 1944, "NaNbO3": 1080, "NaCl": 216}
CH = 128  # input channels per segment
OUT = 512  # output channels per gated segment and per scalar/gate block
X_SEGMENTS = ((1, 2), (3, 3), (5, 3), (7, 2))  # (dim, copies) of the message's irreps 0e, 1o, 2e, 3o
Y_SEGMENTS = ((1, 1), (3, 1))  # node_feats: 0e, 1o
N0 = 4 * OUT  # 0e outputs: 512 scalars, then 3 x 512 gates
C_S, C_G = 1.6790, 1.8462  # normalize2mom constants (arbitrary for timing)


def layout(segments):
    """[(dim, [offsets of its copies])], total width (ir_mul: copy c of a dim-D segment is D * CH columns)."""
    out, at = [], 0
    for dim, copies in segments:
        offsets = []
        for _ in range(copies):
            offsets.append(at)
            at += dim * CH
        out.append((dim, offsets))
    return out, at


@cache
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _acc(acc, x_ptr, xo_ptr, x_w, x_k, y_ptr, yo_ptr, y_w, y_k, w_ptr, N, atom, amask, j, jmask, inv_r, m,
             CHN: tl.constexpr, BLOCK_K: tl.constexpr):
        # message K range: x_k // CHN pieces of CHN channels at offsets xo[p]; row (atom, m) of piece p at
        # atom * x_w + xo[p] + m * CHN. One piece at a time, so no per-element offset gather.
        kk = tl.arange(0, BLOCK_K)
        part = tl.zeros_like(acc)
        for p in range(0, x_k // CHN):
            base = atom * x_w + tl.load(xo_ptr + p) + m * CHN
            for start in range(0, CHN, BLOCK_K):
                a = tl.load(x_ptr + base[:, None] + (start + kk)[None, :], mask=amask[:, None], other=0.0)
                b = tl.load(w_ptr + (p * CHN + start + kk)[:, None] * N + j[None, :], mask=jmask[None, :], other=0.0)
                part += tl.dot(a, b, input_precision="ieee")
        acc += part * inv_r[:, None]
        for p in range(0, y_k // CHN):
            base = atom * y_w + tl.load(yo_ptr + p) + m * CHN
            for start in range(0, CHN, BLOCK_K):
                a = tl.load(y_ptr + base[:, None] + (start + kk)[None, :], mask=amask[:, None], other=0.0)
                b = tl.load(w_ptr + (x_k + p * CHN + start + kk)[:, None] * N + j[None, :], mask=jmask[None, :],
                            other=0.0)
                acc += tl.dot(a, b, input_precision="ieee")
        return acc

    @triton.jit
    def scalars(x_ptr, xo_ptr, x_w, x_k, y_ptr, yo_ptr, y_w, y_k, w_ptr, r_ptr, s_ptr, g_ptr, atoms_n, N, S,
                c_s, c_g, beta, alpha, CHN: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
        pa, pn = tl.program_id(0), tl.program_id(1)
        atom = (pa * BLOCK_A + tl.arange(0, BLOCK_A)).to(tl.int64)
        j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        amask, jmask = atom < atoms_n, j < N
        inv_r = 1.0 / (tl.load(r_ptr + atom, mask=amask, other=1.0) * beta + alpha)
        acc = tl.zeros((BLOCK_A, BLOCK_N), dtype=tl.float32)
        acc = _acc(acc, x_ptr, xo_ptr, x_w, x_k, y_ptr, yo_ptr, y_w, y_k, w_ptr, N, atom, amask, j, jmask, inv_r, 0,
                   CHN, BLOCK_K)
        sig = 1.0 / (1.0 + tl.exp(-acc))
        tile = amask[:, None] & jmask[None, :]
        tl.store(s_ptr + atom[:, None] * S + j[None, :], acc * sig * c_s, mask=tile & (j < S)[None, :])
        tl.store(g_ptr + atom[:, None] * (N - S) + (j - S)[None, :], sig * c_g, mask=tile & (j >= S)[None, :])

    @triton.jit
    def gated(x_ptr, xo_ptr, x_w, x_k, y_ptr, yo_ptr, y_w, y_k, w_ptr, r_ptr, gate_ptr, gate_w, out_ptr, atoms_n,
              N, beta, alpha, D: tl.constexpr, CHN: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr):
        pa, pn = tl.program_id(0), tl.program_id(1)
        atom = (pa * BLOCK_A + tl.arange(0, BLOCK_A)).to(tl.int64)
        j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        amask, jmask = atom < atoms_n, j < N
        tile = amask[:, None] & jmask[None, :]
        inv_r = 1.0 / (tl.load(r_ptr + atom, mask=amask, other=1.0) * beta + alpha)
        gate = tl.load(gate_ptr + atom[:, None] * gate_w + j[None, :], mask=tile, other=0.0)
        for m in tl.static_range(D):
            acc = tl.zeros((BLOCK_A, BLOCK_N), dtype=tl.float32)
            acc = _acc(acc, x_ptr, xo_ptr, x_w, x_k, y_ptr, yo_ptr, y_w, y_k, w_ptr, N, atom, amask, j, jmask, inv_r,
                       m, CHN, BLOCK_K)
            row = atom * D + m
            tl.store(out_ptr + row[:, None] * N + j[None, :], acc * gate, mask=tile)

    return scalars, gated


def operands(atoms, gen, dev):
    xs, x_w = layout(X_SEGMENTS)
    ys, y_w = layout(Y_SEGMENTS)
    ops = {"atoms": atoms, "x": torch.randn(atoms, x_w, generator=gen).to(dev),
           "y": torch.randn(atoms, y_w, generator=gen).to(dev),
           "density": torch.rand(atoms, generator=gen).to(dev) * 30, "beta": 0.07, "alpha": 1.3,
           "x_layout": xs, "y_layout": ys, "weights": [], "x_offsets": [], "y_offsets": []}
    y_by_dim = dict(ys)
    for dim, x_offsets in xs:
        y_offsets = y_by_dim.get(dim, [])
        k = CH * (len(x_offsets) + len(y_offsets))
        n = N0 if dim == 1 else OUT
        ops["weights"].append((torch.randn(k, n, generator=gen) / k ** 0.5).to(dev))
        ops["x_offsets"].append(torch.tensor(x_offsets, dtype=torch.int64, device=dev))
        ops["y_offsets"].append(torch.tensor(y_offsets or [0], dtype=torch.int64, device=dev))
    return ops


def chain(x, y, density, ops):
    """Today's forward (compiled): divide, gather and concatenate the rows, cuBLAS products, the gate."""
    n = x.shape[0]
    x = x / (density * ops["beta"] + ops["alpha"]).unsqueeze(-1)
    y_by_dim = dict(ops["y_layout"])
    outputs = []
    for (dim, x_offsets), w in zip(ops["x_layout"], ops["weights"]):
        pieces = [x[:, o:o + dim * CH].reshape(n * dim, CH) for o in x_offsets]
        pieces += [y[:, o:o + dim * CH].reshape(n * dim, CH) for o in y_by_dim.get(dim, [])]
        outputs.append(torch.cat(pieces, dim=1) @ w)
    s, g = outputs[0][:, :OUT], outputs[0][:, OUT:]
    rows = [torch.nn.functional.silu(s) * C_S]
    gates = torch.sigmoid(g) * C_G
    for i, (dim, _) in enumerate(ops["x_layout"][1:]):
        rows.append((outputs[i + 1].view(n, dim, OUT) * gates[:, i * OUT:(i + 1) * OUT].unsqueeze(1)).reshape(-1, OUT))
    return rows, gates


def fused_fn(ops, config):
    import triton

    block_a, block_n, block_k, warps, stages = config
    atoms, dev = ops["atoms"], ops["x"].device
    x, y = ops["x"], ops["y"]
    s = torch.empty(atoms, OUT, device=dev)
    g = torch.empty(atoms, N0 - OUT, device=dev)
    rows = [torch.empty(atoms * dim, OUT, device=dev) for dim, _ in ops["x_layout"][1:]]
    scalars, gated = _kernels()
    y_by_dim = dict(ops["y_layout"])

    def run():
        dim, x_offsets = ops["x_layout"][0]
        w = ops["weights"][0]
        y_k = CH * len(y_by_dim.get(dim, []))
        scalars[(triton.cdiv(atoms, block_a), triton.cdiv(N0, block_n))](
            x, ops["x_offsets"][0], x.shape[1], CH * len(x_offsets), y, ops["y_offsets"][0], y.shape[1], y_k, w,
            ops["density"], s, g, atoms, N0, OUT, C_S, C_G, ops["beta"], ops["alpha"], CHN=CH, BLOCK_A=block_a,
            BLOCK_N=block_n, BLOCK_K=block_k, num_warps=warps, num_stages=stages)
        for i, (dim, x_offsets) in enumerate(ops["x_layout"][1:]):
            y_k = CH * len(y_by_dim.get(dim, []))
            gated[(triton.cdiv(atoms, block_a), triton.cdiv(OUT, block_n))](
                x, ops["x_offsets"][i + 1], x.shape[1], CH * len(x_offsets), y, ops["y_offsets"][i + 1], y.shape[1],
                y_k, ops["weights"][i + 1], ops["density"], g.reshape(-1)[i * OUT:], g.shape[1], rows[i], atoms, OUT,
                ops["beta"], ops["alpha"], D=dim, CHN=CH, BLOCK_A=block_a, BLOCK_N=block_n, BLOCK_K=block_k,
                num_warps=warps, num_stages=stages)
        return [s, *rows], g
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


CONFIGS = ((32, 64, 32, 4, 3), (64, 64, 32, 4, 3), (32, 128, 32, 4, 3), (64, 128, 32, 8, 3))


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
    compiled = torch.compile(chain, dynamic=False)

    def flush():
        flush_buf.fill_(1.0)

    for name in args.cell or ["NbCl5", "NaNbO3", "NaCl"]:
        atoms = CELLS[name]
        ops = operands(atoms, torch.Generator().manual_seed(20261029), dev)
        record = {"cell": name, "atoms": atoms, "label": args.label, "gpu": torch.cuda.get_device_name(dev),
                  "capability": "sm_%d%d" % torch.cuda.get_device_capability(dev), "torch": torch.__version__}

        def chain_run(ops=ops):
            return compiled(ops["x"], ops["y"], ops["density"], ops)

        ref_rows, ref_gates = chain_run()
        quick, errors = {}, {}
        for config in CONFIGS:
            fn = fused_fn(ops, config)
            rows, gates = fn()
            torch.cuda.synchronize()
            key = "%dx%dx%d" % config[:3]
            errors[key] = {"rows_rel": max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(rows, ref_rows)),
                           "gates_rel": float((gates - ref_gates).abs().max() / ref_gates.abs().max())}
            quick[config] = min(replay_ms(graph_of(fn, flush, args.calls), args.calls) for _ in range(3))
        config = min(quick, key=quick.get)
        record.update(config="%dx%dx%d" % config[:3], errors=errors,
                      quick_ms={"%dx%dx%d" % c[:3]: v for c, v in quick.items()})
        fns = {"chain": chain_run, "fused": fused_fn(ops, config), "flush": lambda: None}
        graphs = {k: graph_of(fn, flush, args.calls) for k, fn in fns.items()}
        times = {k: [] for k in graphs}
        for _ in range(args.blocks):
            for k, g in graphs.items():
                times[k].append(replay_ms(g, args.calls))
        flush_ms = statistics.median(times.pop("flush"))
        for k, v in times.items():
            record[f"{k}_ms"] = statistics.median(v) - flush_ms
            record[f"{k}_spread"] = (max(v) - min(v)) / statistics.median(v)
        record["saved_ms"] = record["chain_ms"] - record["fused_ms"]
        record["fused_over_chain"] = record["fused_ms"] / record["chain_ms"]
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
