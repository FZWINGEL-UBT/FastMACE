"""Time the parts of ``SkinNeighbourList.build_gpu`` (screen, float64 decision, sort, fill) on one GPU.

Neighbour-list follow-up (6 November 2026): the H200 step breakdown of SrTiO3 16×16×16 showed the rebuild growing
5.3× per call for 4.1× the atoms. This splits one build into its parts with CUDA events, for a list of cell
repeats, and times the whole ``build_gpu`` as the engine calls it.

Usage: python tools/bench/rebuild_parts.py --structure results/loup-a100-20261103/SrTiO3.cif --repeats 10 16
       [--cutoff 6.0] [--rounds 5] [--binned 0 1] [--output parts.json]
``--binned``: ``pair_search.BINNED_SCREEN`` per pass (0: the tile screen, 1: the cell-list screen).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ase.io import read

from fastmace.config import NeighbourConfig
from fastmace.neighbors import pair_search, skin
from fastmace.neighbors.skin import SkinNeighbourList, times_3x3


def timed(fn, rounds):
    """Median milliseconds of ``fn()`` over ``rounds`` calls (CUDA events; one warm-up call)."""
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(stop))
    out.sort()
    return out[len(out) // 2]


def parts(atoms, cutoff, rounds):
    device = torch.device("cuda")
    nl = SkinNeighbourList(atoms.cell.array, cutoff, device, torch.float32, NeighbourConfig(),
                           sort_receivers=True, gpu_build=True)
    positions = torch.as_tensor(atoms.positions, dtype=torch.float64, device=device)
    natoms = positions.shape[0]
    r = nl.cutoff + nl.skin
    nl.build_gpu(positions)  # allocate, warm the kernels
    constants = nl._constants(device)
    cell, inverse = constants["cell"], constants["inverse"]
    images, image_vectors = constants["images"], constants["image_vectors"]
    offsets = torch.floor(times_3x3(positions, inverse))
    wrapped = positions - times_3x3(offsets, cell)
    state = nl._pair_search_state
    codes = pair_search.screen_codes(wrapped, image_vectors, r, state, cell=nl.cell)
    found = codes.numel()

    def decide():
        key = torch.empty(found, dtype=torch.int64, device=device)
        kept = torch.zeros((), dtype=torch.int64, device=device)
        for start in range(0, found, skin.BUILD_CHUNK):
            stop = min(found, start + skin.BUILD_CHUNK)
            image, sender, receiver = pair_search.decode_codes(codes[start:stop], natoms)
            vectors = wrapped[receiver] - wrapped[sender] + image_vectors[image]
            unit_shifts = images[image] - offsets[receiver] + offsets[sender]
            keep = ((vectors * vectors).sum(dim=1) < r * r) & ~(
                (sender == receiver) & (unit_shifts == 0).all(dim=1))
            key[start:stop] = torch.where(keep, nl._sort_key(image, sender, receiver, natoms, constants),
                                          torch.iinfo(torch.int64).max)
            kept += keep.sum()
        return key, kept

    key, kept = decide()
    n = int(kept)

    def fill():
        order = torch.sort(key).indices[:n]
        for start in range(0, n, skin.BUILD_CHUNK):
            stop = min(n, start + skin.BUILD_CHUNK)
            image, sender, receiver = pair_search.decode_codes(codes[order[start:stop]], natoms)
            unit_shifts = images[image] - offsets[receiver] + offsets[sender]
            nl.candidate_index[0, start:stop].copy_(sender)
            nl.candidate_index[1, start:stop].copy_(receiver)
            nl.candidate_unit_shifts[start:stop].copy_(unit_shifts)

    def build():
        nl.build_gpu(positions)

    host = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nl.build_gpu(positions)
        torch.cuda.synchronize()
        host.append((time.perf_counter() - t0) * 1e3)
    host.sort()
    return {
        "atoms": natoms, "images": len(images), "screened": found, "candidates": n,
        "ms": {
            "screen": timed(lambda: pair_search.screen_codes(wrapped, image_vectors, r, state, cell=nl.cell), rounds),
            "decide": timed(decide, rounds),
            "sort": timed(lambda: torch.sort(key), rounds),
            "sort_and_fill": timed(fill, rounds),
            "build_gpu": timed(build, rounds),
            "build_gpu_wall": host[len(host) // 2],
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--repeats", type=int, nargs="+", default=[10, 16])
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--binned", type=int, nargs="+", default=[0, 1])
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    base = read(args.structure)
    out = {"gpu": torch.cuda.get_device_name(0), "cutoff": args.cutoff, "results": {}}
    for binned in args.binned:
        pair_search.BINNED_SCREEN = bool(binned)
        for n in args.repeats:
            res = parts(base.repeat((n, n, n)), args.cutoff, args.rounds)
            out["results"][f"binned{binned}-{n}x{n}x{n}"] = res
            print(f"binned={binned} {n}x{n}x{n}", json.dumps(res), flush=True)
    if args.output:
        args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
