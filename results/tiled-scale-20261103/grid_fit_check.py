"""The grids fit_grid tries for SrTiO3 supercells (CPU, no model): per candidate the blocks per axis, the blocks,
the pieces of at most R atoms and the largest block; then fit_grid's choice. grid_fit_check.py R REACH LAYERS NxNxN..."""

import sys

import numpy as np
import torch

from fastmace.model import tiled

R, reach, layers = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
a = 3.905
basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])  # Sr, Ti, O3
for spec in sys.argv[4:]:
    m = int(spec.split("x")[0])
    grid = np.stack(np.meshgrid(*(np.arange(m),) * 3, indexing="ij"), -1).reshape(-1, 1, 3)
    pos = torch.from_numpy(((grid + basis[None]) * a).reshape(-1, 3))
    cell = torch.eye(3, dtype=torch.float64) * a * m
    n = pos.shape[0]
    print(spec, n, flush=True)
    seen = set()
    for scale in tiled.GRID_FIT_SCALES:
        k, skin = tiled._grid_shape(cell, n, R, reach, layers, scale=scale)
        if tuple(k) in seen:
            continue
        seen.add(tuple(k))
        counts, _ = tiled._blocks(pos, cell, k, skin, counts_only=True)
        pieces = sum(-(-c // R) for c in counts)
        print(f"  scale {scale}: k {k} skin {skin > 0} blocks {len(counts)} pieces {pieces} max {max(counts)}",
              flush=True)
    print("  fit_grid:", tiled.fit_grid(pos, cell, R, reach, layers), flush=True)
    del pos
