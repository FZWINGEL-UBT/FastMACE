"""ideas D2 falsifier: per-piece "w-tables" against today's float64 folded products (plans/ideas-20261027.md).

Today the second MH-1 interaction's per-edge weights are ``w = a W^T``, with ``a [E, 65]`` from the certified
Chebyshev tables and ``W [1280, 65]``; the backward's input gradient is ``da = dw W``. Within one table piece,
``a(r) = sum_k T_k(x) C_k`` (``C [terms, 65]``), so ``w(r) = sum_k T_k(x) G_k`` with ``G = C W^T [terms, 1280]``:
17 multiply-adds per output in float64 instead of 65. The backward ``dr = sum_c dw_c w_c'(r)`` uses the derivative
series ``G' [terms - 1, 1280]`` in the same way. ``G`` per (pair, piece) is 174 KB in float64, more than the L2
holds for all pieces, so the kernels take the edges sorted by (pair, piece) and write and read the ``w``/``dw``
rows at their receiver-ordered positions (``perm``).

Synthetic operands with MH-1's shapes (NbCl5 3x3x3: 56,320 padded edges, 4 pairs; NaNbO3: 84,072 edges, 9 pairs),
random coefficients; timing only, plus an exactness check of the kernels against a float64 reference on CPU rows.
The baseline products run through cuBLAS as the float64 package does: set the package's emulation environment
(CUBLAS_EMULATE_DOUBLE_PRECISION=1, CUBLAS_EMULATION_STRATEGY=eager, optionally the mantissa bits) for the process.

Usage: python ideas_d2_wtable.py [--edges 56320] [--pairs 4] [--pieces 366] [--terms 17] -> one JSON line
"""

import argparse
import json
import os

import torch


def kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def forward(x_ptr, group_ptr, perm_ptr, g_ptr, w_ptr, E, Q, TERMS: tl.constexpr, BLOCK_E: tl.constexpr,
                BLOCK_Q: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        rows, cols = e < E, q < Q
        mask = rows[:, None] & cols[None, :]
        x = tl.load(x_ptr + e, mask=rows, other=0.0)
        group = tl.load(group_ptr + e, mask=rows, other=0).to(tl.int64)
        dest = tl.load(perm_ptr + e, mask=rows, other=0).to(tl.int64)
        base = group * TERMS * Q
        t0 = tl.full((BLOCK_E,), 1.0, x.dtype)
        t1 = x
        acc = t0[:, None] * tl.load(g_ptr + base[:, None] + q[None, :], mask=mask, other=0.0)
        acc += t1[:, None] * tl.load(g_ptr + base[:, None] + Q + q[None, :], mask=mask, other=0.0)
        for k in tl.static_range(2, TERMS):
            t0, t1 = t1, 2.0 * x * t1 - t0
            acc += t1[:, None] * tl.load(g_ptr + base[:, None] + k * Q + q[None, :], mask=mask, other=0.0)
        tl.store(w_ptr + dest[:, None] * Q + q[None, :], acc, mask=mask)

    @triton.jit
    def backward(x_ptr, group_ptr, perm_ptr, dg_ptr, dw_ptr, out_ptr, E, Q, TERMS: tl.constexpr,
                 BLOCK_E: tl.constexpr, BLOCK_Q: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        rows = e < E
        x = tl.load(x_ptr + e, mask=rows, other=0.0)
        group = tl.load(group_ptr + e, mask=rows, other=0).to(tl.int64)
        src = tl.load(perm_ptr + e, mask=rows, other=0).to(tl.int64)
        base = group * TERMS * Q
        total = tl.zeros((BLOCK_E,), dtype=x.dtype)
        for start in range(0, Q, BLOCK_Q):
            q = start + tl.arange(0, BLOCK_Q)
            mask = rows[:, None] & (q < Q)[None, :]
            t0 = tl.full((BLOCK_E,), 1.0, x.dtype)
            t1 = x
            d = t0[:, None] * tl.load(dg_ptr + base[:, None] + q[None, :], mask=mask, other=0.0)
            d += t1[:, None] * tl.load(dg_ptr + base[:, None] + Q + q[None, :], mask=mask, other=0.0)
            for k in tl.static_range(2, TERMS):
                t0, t1 = t1, 2.0 * x * t1 - t0
                d += t1[:, None] * tl.load(dg_ptr + base[:, None] + k * Q + q[None, :], mask=mask, other=0.0)
            g = tl.load(dw_ptr + src[:, None] * Q + q[None, :], mask=mask, other=0.0)
            total += tl.sum(g * d, axis=1)
        tl.store(out_ptr + e, total, mask=rows)

    return forward, backward


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edges", type=int, default=56320)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--pieces", type=int, default=366)
    p.add_argument("--terms", type=int, default=17)
    p.add_argument("--q", type=int, default=1280)
    p.add_argument("--inputs", type=int, default=65)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--configs", default="16x64,32x64,16x128,32x32")
    args = p.parse_args()
    import triton

    torch.backends.cuda.preferred_blas_library("cublas")
    dev, f64 = "cuda", torch.float64
    gen = torch.Generator().manual_seed(1027)
    E, Q, P, T = args.edges, args.q, args.inputs, args.terms
    groups = args.pairs * args.pieces
    # Edges sorted by (pair, piece): each touched group gets a run of edges; perm scatters them to receiver order.
    group = torch.sort(torch.randint(0, groups, (E,), generator=gen)).values.to(torch.int32)
    x = torch.rand(E, generator=gen, dtype=f64) * 2 - 1
    perm = torch.randperm(E, generator=gen).to(torch.int32)
    C = torch.randn(groups, T, P, generator=gen, dtype=f64) / T
    W = torch.randn(Q, P, generator=gen, dtype=f64) / P ** 0.5
    G = (C @ W.T).contiguous()  # [groups, T, Q]
    Gd = torch.randn(groups, T, Q, generator=gen, dtype=f64) / T  # derivative series (stand-in values)
    xg, groupg, permg = x.to(dev), group.to(dev), perm.to(dev)
    Gg, Gdg, Wg = G.to(dev), Gd.to(dev), W.to(dev)
    a = torch.randn(E, P, generator=gen, dtype=f64).to(dev)
    dw = torch.randn(E, Q, generator=gen, dtype=f64).to(dev)
    w = torch.empty(E, Q, dtype=f64, device=dev)
    dr = torch.empty(E, dtype=f64, device=dev)
    fwd, bwd = kernels()
    out = {"edges": E, "pairs": args.pairs, "pieces": args.pieces, "terms": T, "gpu": torch.cuda.get_device_name(),
           "env": {k: v for k, v in os.environ.items() if k.startswith("CUBLAS")},
           "G_MB": Gg.numel() * 8 / 1e6}
    out["gemm_forward_ms"] = timed(lambda: a @ Wg.T, args.reps)
    out["gemm_input_grad_ms"] = timed(lambda: dw @ Wg, args.reps)
    best_f = best_b = None
    for cfg in args.configs.split(","):
        be, bq = (int(v) for v in cfg.split("x"))
        tf = timed(lambda: fwd[(triton.cdiv(E, be), triton.cdiv(Q, bq))](xg, groupg, permg, Gg, w, E, Q, TERMS=T,
                                                                          BLOCK_E=be, BLOCK_Q=bq), args.reps)
        tb = timed(lambda: bwd[(triton.cdiv(E, be),)](xg, groupg, permg, Gdg, dw, dr, E, Q, TERMS=T, BLOCK_E=be,
                                                      BLOCK_Q=bq), args.reps)
        out[f"wtable_forward_ms_{cfg}"], out[f"wtable_backward_ms_{cfg}"] = tf, tb
        best_f = tf if best_f is None else min(best_f, tf)
        best_b = tb if best_b is None else min(best_b, tb)
    out["wtable_forward_ms"], out["wtable_backward_ms"] = best_f, best_b
    # Exactness on sampled rows against a CPU float64 reference.
    be, bq = 16, 64
    fwd[(triton.cdiv(E, be), triton.cdiv(Q, bq))](xg, groupg, permg, Gg, w, E, Q, TERMS=T, BLOCK_E=be, BLOCK_Q=bq)
    bwd[(triton.cdiv(E, be),)](xg, groupg, permg, Gdg, dw, dr, E, Q, TERMS=T, BLOCK_E=be, BLOCK_Q=bq)
    torch.cuda.synchronize()
    rows = torch.randperm(E, generator=gen)[:256]
    cheb = torch.empty(len(rows), T, dtype=f64)
    cheb[:, 0], cheb[:, 1] = 1.0, x[rows]
    for k in range(2, T):
        cheb[:, k] = 2 * x[rows] * cheb[:, k - 1] - cheb[:, k - 2]
    a_rows = torch.einsum("et,etp->ep", cheb, C[group[rows].long()])
    w_ref = a_rows @ W.T  # today's chain: a, then a W^T
    w_got = w.cpu()[perm[rows].long()]
    scale = w_ref.abs().amax(1, keepdim=True)
    out["forward_max_rel_err_vs_chain"] = float(((w_got - w_ref).abs() / scale).max())
    d_ref = torch.einsum("et,etq->eq", cheb, Gd[group[rows].long()])
    dr_ref = (dw.cpu()[perm[rows].long()] * d_ref).sum(1)
    out["backward_max_rel_err"] = float(((dr.cpu()[rows] - dr_ref).abs()
                                         / (dw.cpu()[perm[rows].long()].abs() * d_ref.abs()).sum(1)).max())
    print(json.dumps(out))


if __name__ == "__main__":
    main()
