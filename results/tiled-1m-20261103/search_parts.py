"""Where a tiled list rebuild's time goes at 1M SrTiO3 (58^3, cutoff 6 + skin 0.3): binning, the per-tile 27-bin
search (candidates, distance test, sort), the kept lists' packing and host copies, and the plan. Device timings
with synchronisation per phase; OMAT-0 medium's cutoff and the MD run's skin."""
import json
import sys
import time

import torch
from ase import Atoms

from fastmace.model.tiled import _narrow_counts, _pack_senders, _pack_units, plan_tiles
from fastmace.neighbors.tile_list import TileNeighbours

device = torch.device("cuda")

a = 3.905
unit = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (.5, .5, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)],
             cell=[a, a, a], pbc=True)
atoms = unit.repeat((58, 58, 58))
rng = torch.Generator().manual_seed(1)
pos = torch.as_tensor(atoms.positions, dtype=torch.float64)
pos = (pos + 0.05 * torch.randn(pos.shape, generator=rng, dtype=torch.float64)).to(device)
cell = torch.as_tensor(atoms.cell.array, dtype=torch.float64)
cutoff = 6.0 + 0.3


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


times = {}
for repeat in range(3):
    t0 = sync()
    tiles, grid = plan_tiles(pos, cell.to(device), None, 16384, 2**62)
    t1 = sync()
    bins = TileNeighbours(cutoff, cell, torch.float32, device=device)
    bins.bin(pos)
    t2 = sync()
    search = pack = copy = 0.0
    edges = 0
    for receivers in tiles:
        s0 = sync()
        sender, receiver, unit_shift, shifts, inner = bins.edges(receivers, inner=6.0)
        s1 = sync()
        packed = _pack_units(unit_shift)
        counts = torch.bincount(torch.searchsorted(receivers, receiver), minlength=receivers.shape[0])
        entry = (*_pack_senders(sender), _narrow_counts(counts), packed)
        s2 = sync()
        host = tuple(t.cpu() for t in entry)
        s3 = sync()
        search += s1 - s0
        pack += s2 - s1
        copy += s3 - s2
        edges += sender.shape[0]
        del host
    times[repeat] = dict(plan=t1 - t0, bin=t2 - t1, search=search, pack=pack, host_copy=copy, edges=edges,
                         tiles=len(tiles))
print(json.dumps(dict(device=torch.cuda.get_device_name(), atoms=len(atoms), cutoff=cutoff, times=times), indent=1))
sys.stdout.flush()
