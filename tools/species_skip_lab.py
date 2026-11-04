"""Species skip lab (4 November 2026, Benchmark_loup speed goal): what grouping the skip connection's rows by species
could save, on Benchmark_loup's shapes (OMAT-0 medium, SrTiO3 10x10x10: 5,000 atoms, 1,000 Sr, 1,000 Ti, 3,000 O).

``SpeciesSkip`` multiplies every atom's rows by the concatenated [128, species x 128] matrices of all present species
and keeps its own block (y * onehot summed over species); autograd expands the gradient to all species and multiplies
it back. Timed here, forward plus input backward, CUDA events, median, variants alternating in one process:

- ``current``: SpeciesSkip's product (``tf32x3_matmul`` with ``--tf32x3``, else ``@``);
- ``grouped``: atoms sorted by species once, one product per species on its rows (index_select / index_copy), the
  bound of an exact grouped kernel (eager, data-dependent shapes; a production kernel needs fixed shapes);
- ``bound`` (wrong result): every row through one species' matrix, no gather: what a grouped GEMM that gathers its
  rows in the prologue and scatters in the epilogue could cost at best.

Interaction 0: rows of 0e+1o+2e+3o (16 per atom), interaction 1: 0e (1 per atom); 128 channels in and out.

Usage (hybrid environment, FastMACE on PYTHONPATH): python tools/species_skip_lab.py --output DIR [--tf32x3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)
import torch

COUNTS = (1000, 1000, 3000)
SIZES = (16, 1)
MUL = 128


def product(tf32x3):
    if tf32x3:
        from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul
        return tf32x3_matmul
    return torch.matmul


def make(size, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    species = torch.cat([torch.full((n,), s, device="cuda") for s, n in enumerate(COUNTS)])
    species = species[torch.randperm(species.numel(), generator=g, device="cuda")]
    atoms = species.numel()
    onehot = torch.nn.functional.one_hot(species, len(COUNTS)).float()
    x = torch.randn(atoms, size, MUL, device="cuda", generator=g, requires_grad=True)
    w = torch.randn(len(COUNTS), MUL, MUL, device="cuda", generator=g) / MUL ** 0.5
    wcat = w.permute(1, 0, 2).reshape(MUL, len(COUNTS) * MUL).contiguous()
    grad = torch.randn(atoms, size, MUL, device="cuda", generator=g)
    return species, onehot, x, w, wcat, grad


def current(mm, x, onehot, wcat):
    atoms, size, _ = x.shape
    y = mm(x.reshape(atoms * size, MUL), wcat).view(atoms, size, len(COUNTS), MUL)
    return (y * onehot[:, None, :, None]).sum(dim=2)


def grouped(mm, x, groups, w):
    atoms, size, _ = x.shape
    out = x.new_empty(atoms, size, MUL)
    for s, rows in enumerate(groups):
        part = mm(x.index_select(0, rows).reshape(-1, MUL), w[s]).view(-1, size, MUL)
        out = out.index_copy(0, rows, part)
    return out


def bound(mm, x, w):
    atoms, size, _ = x.shape
    return mm(x.reshape(atoms * size, MUL), w[0]).view(atoms, size, MUL)


def timed(fn, x, grad, rounds):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        x.grad = None
        torch.autograd.backward(fn(), grad)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / rounds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tf32x3", action="store_true", help="the products through tf32x3_matmul (the A100 plan)")
    p.add_argument("--blocks", type=int, default=15)
    p.add_argument("--rounds", type=int, default=20)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    mm = product(args.tf32x3)
    result = dict(gpu=torch.cuda.get_device_name(), tf32x3=args.tf32x3, counts=COUNTS, sizes={})
    for size in SIZES:
        species, onehot, x, w, wcat, grad = make(size)
        groups = [torch.nonzero(species == s).flatten() for s in range(len(COUNTS))]
        fns = {"current": lambda: current(mm, x, onehot, wcat), "grouped": lambda: grouped(mm, x, groups, w),
               "bound": lambda: bound(mm, x, w)}
        # exactness: the same per-element dot products
        x.grad = None
        torch.autograd.backward(fns["current"](), grad)
        ref_y, ref_g = fns["current"]().detach(), x.grad.clone()
        x.grad = None
        torch.autograd.backward(fns["grouped"](), grad)
        diff = dict(y=float((fns["grouped"]().detach() - ref_y).abs().max()), dx=float((x.grad - ref_g).abs().max()))
        for fn in fns.values():
            timed(fn, x, grad, 3)
        times = {name: [] for name in fns}
        for block in range(args.blocks):
            order = list(fns) if block % 2 == 0 else list(reversed(fns))
            for name in order:
                times[name].append(timed(fns[name], x, grad, args.rounds))
        med = {name: sorted(t)[len(t) // 2] for name, t in times.items()}
        result["sizes"][size] = dict(median_us=med, ratio=med["grouped"] / med["current"],
                                     bound_ratio=med["bound"] / med["current"], max_abs_diff=diff,
                                     times_us=times)
        print(f"size {size:2d}: current {med['current']:8.1f} us  grouped {med['grouped']:8.1f} us  "
              f"bound {med['bound']:8.1f} us  ratios {med['grouped'] / med['current']:.3f} "
              f"{med['bound'] / med['current']:.3f}  diff {diff}", flush=True)
    (args.output / "species_skip_lab.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
