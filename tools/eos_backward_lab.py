"""Round 2, outside idea 2 (5 November 2026): the ``edge_outer_sum`` backward on Benchmark_loup's shapes.

The first interaction's per-slot outer-product sums take their backward in ``edge_outer_sum_backward``: per edge,
grad_a = sum_d G[key, d, :] y[d] and grad_y = sum_k G[key, :, k] a[k], with G the slot's [D, K] gradient block. This
times, on the real key order (SrTiO3 10x10x10, a jittered lattice, edges sorted by receiver and then sender species,
key = receiver * species + sender species) and random values:
- ``base``: the plan's kernel (one gathered G block per edge, 32 edges per program);
- ``F-one-block``: the base kernel with every key 0 (one G block, always in L1): the bound of perfect sharing;
- ``F-const-G``: the base kernel without G loads (G = 1): the bound without any gather;
- ``walk-c{chunk}-w{warps}``: ``BACKWARD_WALK`` (one G load per run of equal keys);
- ``copy``: the DRAM traffic of a, y, grad_a and grad_y alone (two copies), the floor without G;
- ``table-bwd``, ``base+table-bwd``, ``walk+table-bwd``, ``walk-dr``: the table backward (dr from grad_a and the
  derivative table) separate, after either backward, or fused into the walk (``dr_kernels``).
Accuracy of every variant that computes the real result: max |d| / max |ref| against a float64 reference.

Usage: python tools/eos_backward_lab.py --output DIR
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)


def keys_of(structure, cutoff, jitter, seed=0):
    import numpy as np
    from ase.io import read
    from ase.neighborlist import neighbor_list

    path, _, repeat = structure.partition("@")
    atoms = read(path)
    if repeat:
        atoms = atoms.repeat(tuple(int(v) for v in repeat.split("x")))
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0.0, jitter, atoms.positions.shape)
    i, j = neighbor_list("ij", atoms, cutoff)
    species = np.unique(atoms.numbers, return_inverse=True)[1]
    slots = int(species.max()) + 1
    order = np.lexsort((species[i], j))  # receiver, then sender species
    key = j[order] * slots + species[i[order]]
    return key, len(atoms) * slots


def const_g_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(grad_ptr, a_ptr, y_ptr, key_ptr, grad_a_ptr, grad_y_ptr, num_edges, a_stride,
               K: tl.constexpr, D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr):
        edges = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        edge_mask = edges < num_edges
        edges = edges.to(tl.int64)
        k = tl.arange(0, BLOCK_K)
        mask = edge_mask[:, None] & (k < K)[None, :]
        key = tl.load(key_ptr + edges, mask=edge_mask, other=0).to(tl.int64)
        a = tl.load(a_ptr + edges[:, None] * a_stride + k[None, :], mask=mask, other=0.0)
        grad_a = a * 0
        for d in tl.static_range(D):
            g = a * 0 + (key[:, None] + d).to(tl.float32)  # no G load
            y = tl.load(y_ptr + edges * D + d, mask=edge_mask, other=0.0)
            grad_a += g * y[:, None]
            tl.store(grad_y_ptr + edges * D + d, tl.sum(g * a, axis=1), mask=edge_mask)
        tl.store(grad_a_ptr + edges[:, None] * K + k[None, :], grad_a, mask=mask)

    return kernel


def dr_kernels(terms):
    """Round-2 idea (5 November 2026): the table backward fused into the walk. ``table_bwd``: dr_e = sum_k grad_a[e, k]
    da_k/dr from the edge's derivative-table row (group g_e, argument t_e: sum_j T_j'(t) D[g, j, k], the separate
    table-backward pass, timing stand-in for the 108.5 us kernel of the A100 profile); ``walk_dr``: the walk with the
    same dr computed from the grad_a row in registers, no grad_a stored."""
    import triton
    import triton.language as tl

    @triton.jit
    def _slope(dtab_ptr, group, t, k, K: tl.constexpr, TERMS: tl.constexpr):
        # sum_j T_j'(t) D[group, j, :] with T_j' = j U_{j-1}
        u_prev = 0.0 * t
        u = 1.0 + 0.0 * t
        out = tl.zeros([k.shape[0]], dtype=tl.float32)
        for j in tl.static_range(1, TERMS):
            row = tl.load(dtab_ptr + (group * TERMS + j) * K + k, mask=k < K, other=0.0)
            out += (j * u) * row
            u_next = 2.0 * t * u - u_prev
            u_prev = u
            u = u_next
        return out

    @triton.jit
    def table_bwd(grad_a_ptr, dtab_ptr, group_ptr, t_ptr, dr_ptr, num_edges, K: tl.constexpr, TERMS: tl.constexpr,
                  BLOCK_K: tl.constexpr):
        e = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, BLOCK_K)
        g = tl.load(group_ptr + e).to(tl.int64)
        t = tl.load(t_ptr + e)
        ga = tl.load(grad_a_ptr + e * K + k, mask=k < K, other=0.0)
        tl.store(dr_ptr + e, tl.sum(ga * _slope(dtab_ptr, g, t, k, K, TERMS), axis=0))

    @triton.jit
    def walk_dr(grad_ptr, a_ptr, y_ptr, key_ptr, dtab_ptr, group_ptr, t_ptr, dr_ptr, grad_y_ptr, num_edges, a_stride,
                K: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
                TERMS: tl.constexpr):
        start = tl.program_id(0) * CHUNK
        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        mask = (d < D)[:, None] & (k < K)[None, :]
        offsets = d[:, None] * K + k[None, :]
        current = tl.load(key_ptr + start).to(tl.int64)
        block = tl.load(grad_ptr + current * (D * K) + offsets, mask=mask, other=0.0)
        stop = tl.minimum(start + CHUNK, num_edges)
        for edge in range(start, stop):
            key = tl.load(key_ptr + edge).to(tl.int64)
            if key != current:
                block = tl.load(grad_ptr + key * (D * K) + offsets, mask=mask, other=0.0)
                current = key
            row = edge.to(tl.int64)
            a = tl.load(a_ptr + row * a_stride + k, mask=k < K, other=0.0)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            ga = tl.sum(block * y[:, None], axis=0)
            g = tl.load(group_ptr + row).to(tl.int64)
            t = tl.load(t_ptr + row)
            tl.store(dr_ptr + row, tl.sum(ga * _slope(dtab_ptr, g, t, k, K, TERMS), axis=0))
            tl.store(grad_y_ptr + row * D + d, tl.sum(block * a[None, :], axis=1), mask=d < D)

    return table_bwd, walk_dr


def main(argv=None):
    import torch
    import triton

    from fastmace.kernels.triton import edge_outer_sum as eos

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--jitter", type=float, default=0.07)
    p.add_argument("--rounds", type=int, default=31)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    key_np, slots = keys_of(args.structure, args.cutoff, args.jitter)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    edges, d, k = len(key_np), 16, 64
    key = torch.as_tensor(key_np, dtype=torch.int64, device=dev)
    table = torch.randn((edges, k + 1), device=dev, generator=gen)
    a = table[:, :k]  # the strided [E, 64] slice of the [E, 65] table rows, as in the plan
    y = torch.randn((edges, d), device=dev, generator=gen)
    grad = torch.randn((slots, d * k), device=dev, generator=gen)
    ref_a = torch.einsum("edk,ed->ek", grad.double().view(-1, d, k)[key], y.double())
    ref_y = torch.einsum("edk,ek->ed", grad.double().view(-1, d, k)[key], a.double())
    zeros = torch.zeros_like(key)
    grad_a, grad_y = torch.empty((edges, k), device=dev), torch.empty_like(y)

    def base(keys):
        return lambda: eos._triton_backward()[(triton.cdiv(edges, eos.BACKWARD_BLOCK_EDGES),)](
            grad, a, y, keys, grad_a, grad_y, edges, a.stride(0), K=k, D=d, INNER=1,
            BLOCK_E=eos.BACKWARD_BLOCK_EDGES, BLOCK_K=64)

    def walk(chunk, warps):
        return lambda: eos._triton_backward_walk()[(triton.cdiv(edges, chunk),)](
            grad, a, y, key, grad_a, grad_y, edges, a.stride(0), K=k, D=d, INNER=1, CHUNK=chunk, BLOCK_K=64,
            BLOCK_D=16, num_warps=warps)

    # The table backward's inputs: Benchmark_loup's interaction-1 table shape (9 pairs x 893 pieces, degree 5:
    # 5 derivative terms), random groups and arguments (an L2-resident 10 MB table, as on the A100).
    terms = 6
    groups = 9 * 893
    dtab = torch.randn((groups, terms, k), device=dev, generator=gen)
    group = torch.randint(0, groups, (edges,), device=dev, generator=gen, dtype=torch.int32)
    t = torch.rand((edges,), device=dev, generator=gen) * 2 - 1
    dr = torch.empty((edges,), device=dev)
    table_bwd, walk_dr = dr_kernels(terms)
    chunk = 16

    def walk_then_table():
        walk(chunk, 1)()
        table_bwd[(edges,)](grad_a, dtab, group, t, dr, edges, K=k, TERMS=terms, BLOCK_K=64)

    def base_then_table():
        base(key)()
        table_bwd[(edges,)](grad_a, dtab, group, t, dr, edges, K=k, TERMS=terms, BLOCK_K=64)

    def fused():
        walk_dr[(triton.cdiv(edges, chunk),)](grad, a, y, key, dtab, group, t, dr, grad_y, edges, a.stride(0), K=k, D=d,
                                              CHUNK=chunk, BLOCK_K=64, BLOCK_D=16, TERMS=terms, num_warps=1)

    const = const_g_kernel()
    variants = {
        "base": (base(key), True),
        "F-one-block": (base(zeros), False),
        "F-const-G": (lambda: const[(triton.cdiv(edges, 32),)](grad, a, y, key, grad_a, grad_y, edges, a.stride(0),
                                                                K=k, D=d, BLOCK_E=32, BLOCK_K=64), False),
        "copy": (lambda: (grad_a.copy_(a), grad_y.copy_(y)), False),
    }
    for chunk_ in (16, 32, 64):
        for warps in (1, 2, 4):
            variants[f"walk-c{chunk_}-w{warps}"] = (walk(chunk_, warps), True)
    variants.update({"table-bwd": (lambda: table_bwd[(edges,)](grad_a, dtab, group, t, dr, edges, K=k, TERMS=terms,
                                                               BLOCK_K=64), False),
                     "base+table-bwd": (base_then_table, False), "walk+table-bwd": (walk_then_table, False),
                     "walk-dr": (fused, False)})
    rows = {}
    for name, (fn, real) in variants.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        if real:
            rows[name] = dict(err_a=float((grad_a.double() - ref_a).abs().max() / ref_a.abs().max()),
                              err_y=float((grad_y.double() - ref_y).abs().max() / ref_y.abs().max()))
        else:
            rows[name] = {}
    walk_then_table()
    torch.cuda.synchronize()
    ref_dr = dr.clone()
    fused()
    torch.cuda.synchronize()
    rows["walk-dr"]["err_dr"] = float((dr - ref_dr).abs().max() / ref_dr.abs().max())
    times = {name: [] for name in variants}
    for _ in range(args.rounds):
        for name, (fn, _) in variants.items():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000.0)
    base_us = sorted(times["base"])[len(times["base"]) // 2]
    for name in variants:
        t = sorted(times[name])
        rows[name].update(median_us=t[len(t) // 2], ratio=t[len(t) // 2] / base_us)
        print(f"{name:14s} {rows[name]['median_us']:8.1f} us  ratio {rows[name]['ratio']:.3f}  "
              f"{ {x: f'{v:.1e}' for x, v in rows[name].items() if x.startswith('err')} }", flush=True)
    runs = torch.unique_consecutive(key, return_counts=True)[1].float()
    meta = dict(tool="eos_backward_lab", created_utc=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                triton=triton.__version__, edges=edges, slots=slots, mean_run=float(runs.mean()),
                structure=args.structure, rounds=args.rounds, variants=rows)
    (args.output / "eos_lab.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
