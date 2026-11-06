"""Idea 38 CPU check: receiver-tile lengths of the w' edge order (SrTiO3 10x10x10, OMAT-0 cutoff 6 A, skin 0.5 A).

Emulates FastMACE's skin list (fastmace/neighbors/skin.py): pairs within cutoff + skin, stably sorted by receiver
(sort_receivers is on for the hybrid graph engine), then compacted stably so the pairs within the cutoff come first,
then padding to the captured capacity (373,760). Tiles are runs of equal receiver in that order, cut into pieces of
at most TILE edges. Reports the share of contributing edges (r < cutoff) in tiles of at least 8 edges (plan stop
line: below one half). Thermal disorder: Gaussian displacements of SIGMA A per Cartesian component.

python tile_census.py CIF [SIGMA]   (WSL ~/environments/mace_cpu: ase, matscipy)
"""
import sys

import numpy as np
from ase.io import read
from matscipy.neighbours import neighbour_list

CUTOFF, SKIN, CAPACITY = 6.0, 0.5, 373_760
atoms = read(sys.argv[1]).repeat((10, 10, 10))
sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.07
rng = np.random.default_rng(20261106)
atoms.positions += rng.normal(0.0, sigma, atoms.positions.shape)
i, j, d = neighbour_list("ijd", atoms, CUTOFF + SKIN)
# MACE convention: edge (sender = j, receiver = i) or the reverse; the multiset of receivers is the same either way.
receiver = i
order = np.argsort(receiver, kind="stable")
receiver, d = receiver[order], d[order]
inside = d < CUTOFF
receiver = np.concatenate([receiver[inside], receiver[~inside]])
contributing = np.concatenate([np.ones(inside.sum(), bool), np.zeros((~inside).sum(), bool)])
pad = CAPACITY - len(receiver)
print(f"atoms {len(atoms)}, pairs within cutoff {inside.sum()}, skin-only {(~inside).sum()}, padding {pad}")
# The w' sees the first CAPACITY compacted edges (all pairs within the cutoff, then skin pairs or padding).
receiver, contributing = receiver[:CAPACITY], contributing[:CAPACITY]
receiver = np.concatenate([receiver, np.zeros(max(pad, 0), int)])
contributing = np.concatenate([contributing, np.zeros(max(pad, 0), bool)])
starts = np.flatnonzero(np.r_[True, receiver[1:] != receiver[:-1]])
lengths = np.diff(np.r_[starts, len(receiver)])
inside_runs = np.add.reduceat(contributing, starts)
print(f"receiver runs {len(starts)}; run length of contributing edges: mean {inside_runs[inside_runs > 0].mean():.1f}, "
      f"min {inside_runs[inside_runs > 0].min()}, max {inside_runs.max()}")
for tile in (16, 32):
    sizes, contrib = [], []
    for s, n in zip(starts, lengths):
        for a in range(0, n, tile):
            m = min(tile, n - a)
            sizes.append(m)
            contrib.append(int(contributing[s + a:s + a + m].sum()))
    sizes, contrib = np.array(sizes), np.array(contrib)
    share = contrib[sizes >= 8].sum() / contrib.sum()
    full = contrib[sizes == tile].sum() / contrib.sum()
    print(f"tile {tile}: {len(sizes)} tiles (mean {sizes.mean():.2f} edges); contributing edges in tiles >= 8: "
          f"{share:.4f}; in full tiles: {full:.4f}; dout rows read per edge {len(sizes) / len(receiver):.4f}")
