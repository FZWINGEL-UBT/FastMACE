"""Offline slab census for PACKED8's structure dependence (packed8-auto, 12 November 2026). CPU, numpy only.

How many distinct table slabs (species pair, spline interval) do the edges of a structure touch, and how many per
tile of 256 consecutive edges (receiver order, each receiver's edges by length, as SORT_BY_LENGTH)? A slab is one
(pair, interval) row of the SPLINE_W table, about 20 KB (Q x 16 bytes). The production grid is the Hermite grid of
results/bspline-reuse-bound-20261109/c2_certificate.json (736 intervals; 0.021 A wide beyond 2.5 A).

Thermal states are modelled as Gaussian position noise sigma(T) = 0.08 A * sqrt(T / 300 K) (harmonic scaling of
an assumed 0.08 A rms displacement at 300 K in SrTiO3); the 0.3 A rattle is the lab's "disordered" snapshot.

    python results/packed8-auto-20261112/slab_census.py > results/packed8-auto-20261112/slab_census.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

GRID = [(0.3, 0.5, 161), (0.5, 0.75, 113), (0.75, 1.0, 87), (1.0, 1.5, 124), (1.5, 2.0, 54), (2.0, 2.5, 31),
        (2.5, 6.0, 166)]
R_MAX = 6.0
SLAB_KB = 20.0
TILE = 256


def interval(r):
    """The spline interval index of lengths ``r`` on GRID."""
    iv = np.zeros(r.shape, dtype=np.int64)
    base = 0
    for lo, hi, n in GRID:
        inside = (r >= lo) & (r < hi)
        iv[inside] = base + np.minimum(((r[inside] - lo) / (hi - lo) * n).astype(np.int64), n - 1)
        base += n
    return iv


def srtio3(n=10, a=3.905):
    """Cubic perovskite SrTiO3, n x n x n cells: positions, species (0 Sr, 1 Ti, 2 O), box length."""
    basis = np.array([[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    species = np.array([0, 1, 2, 2, 2])
    cells = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    pos = ((cells[:, None, :] + basis[None]) * a).reshape(-1, 3)
    return pos, np.tile(species, len(cells)), n * a


def census(pos, species, box, chunk=500):
    n = len(pos)
    recv, length, pair = [], [], []
    for s in range(0, n, chunk):
        d = pos[None, :, :] - pos[s:s + chunk, None, :]
        d -= box * np.round(d / box)
        r = np.sqrt((d * d).sum(-1))
        i, j = np.nonzero((r < R_MAX) & (r > 1e-6))
        recv.append(i + s)
        length.append(r[i, j])
        pair.append(species[i + s] * 3 + species[j])
    recv, length, pair = map(np.concatenate, (recv, length, pair))
    order = np.lexsort((length, recv))
    slab = (pair * 736 + interval(length))[order]
    distinct = np.unique(slab).size
    tiles = [np.unique(slab[k:k + TILE]).size for k in range(0, slab.size, TILE)]
    # within-tile reuse: edges per distinct slab in a tile (high = L1 reuse)
    return dict(edges=int(slab.size), distinct_slabs=int(distinct), footprint_MB=round(distinct * SLAB_KB / 1024, 1),
                slabs_per_tile=round(float(np.mean(tiles)), 1), edges_per_slab_in_tile=round(TILE / float(np.mean(tiles)), 2))


def main():
    pos0, species, box = srtio3()
    rng = np.random.default_rng(1)
    cases = [("perfect lattice", 0.0)] + [(f"{t} K (sigma {0.08 * (t / 300) ** .5:.3f} A)", 0.08 * (t / 300) ** .5)
                                          for t in (1, 10, 50, 300, 1500)] + [("0.3 A rattle (lab disordered)", 0.3)]
    out = {}
    for label, sigma in cases:
        pos = pos0 + rng.normal(0, sigma, pos0.shape) if sigma else pos0
        out[label] = census(pos, species, box)
        print(label, out[label], file=sys.stderr, flush=True)
    print(json.dumps(dict(structure="SrTiO3 10x10x10 (5,000 atoms)", grid_intervals=736, tile=TILE, cases=out),
                     indent=2))


if __name__ == "__main__":
    main()
