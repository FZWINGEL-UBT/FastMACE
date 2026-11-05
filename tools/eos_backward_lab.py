"""Round 2, outside idea 2 (5 November 2026): the ``edge_outer_sum`` backward on Benchmark_loup's shapes.

The first interaction's per-slot outer-product sums take their backward in ``edge_outer_sum_backward``: per edge,
grad_a = sum_d G[key, d, :] y[d] and grad_y = sum_k G[key, :, k] a[k], with G the slot's [D, K] gradient block. This
times, on the real key order (SrTiO3 10x10x10, a jittered lattice, edges sorted by receiver and then sender species,
key = receiver * species + sender species) and random values:
- ``base``: the plan's kernel (one gathered G block per edge, 32 edges per program);
- ``F-one-block``: the base kernel with every key 0 (one G block, always in L1): the bound of perfect sharing;
- ``F-const-G``: the base kernel without G loads (G = 1): the bound without any gather;
- ``walk-c{chunk}-w{warps}``: ``BACKWARD_WALK`` (one G load per run of equal keys);
- ``copy``: the DRAM traffic of a, y, grad_a and grad_y alone (two copies), the floor without G.
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

    const = const_g_kernel()
    variants = {
        "base": (base(key), True),
        "F-one-block": (base(zeros), False),
        "F-const-G": (lambda: const[(triton.cdiv(edges, 32),)](grad, a, y, key, grad_a, grad_y, edges, a.stride(0),
                                                                K=k, D=d, BLOCK_E=32, BLOCK_K=64), False),
        "copy": (lambda: (grad_a.copy_(a), grad_y.copy_(y)), False),
    }
    for chunk in (16, 32, 64):
        for warps in (1, 2, 4):
            variants[f"walk-c{chunk}-w{warps}"] = (walk(chunk, warps), True)
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
