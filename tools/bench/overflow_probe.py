"""Billion-atom plan, step 2 (plans/billion-atoms-20261111.md): the tiled call's whole-system [N, 3] paths above
2^31 elements, on a fabricated structure of ``--atoms`` atoms (default 750,000,000: 3N = 2.25e9 > 2^31).

Above 715,827,883 atoms the flattened [N, 3] arrays have more than 2^31 elements, which the 397.5M record never
reached. A tile reads them through the compiled edge-vector functions (``_tile_vectors`` on the float32 positions,
``_slot_local_vectors`` on the float64 ones, both ``torch.compile(dynamic=False)`` as ``_compiled_once``) and writes
the forces and node energies with ``index_add_``. Each check runs one tile-shaped slot whose rows sit at the start,
in the middle and at the end of the arrays (the end beyond 2^31 / 3), against the same function on a small copy of
those rows (re-indexed): the results must be bitwise equal. Also the whole-array reductions the call makes
(``abs().max()``) and the ``narrow`` view of the forces. Writes ``<output>/overflow-probe.json``; exit 1 on a
mismatch. Needs about 4 x 12 B x N of device memory (36 GB at 750M; Colab G4 or an A100-80).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch)
import torch

from fastmace.model import tiled


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", type=int, default=750_000_000)
    p.add_argument("--rows", type=int, default=24_000, help="slot rows (receivers + halo)")
    p.add_argument("--edges", type=int, default=1_500_000)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def fill(positions64, chunk=1 << 26):
    """Coordinates of a 2,300 A cube on a lattice of the atom index (deterministic, chunk by chunk)."""
    n = positions64.shape[0]
    for start in range(0, n, chunk):
        index = torch.arange(start, min(n, start + chunk), device=positions64.device, dtype=torch.int64)
        positions64[start:start + index.numel(), 0] = (index % 1150).double() * 2.0 + 0.01 * (index % 7).double()
        positions64[start:start + index.numel(), 1] = (index // 1150 % 1150).double() * 2.0 + 0.003 * (index % 5)
        positions64[start:start + index.numel(), 2] = (index // 1322500).double() * 2.0 + 0.007 * (index % 3)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda")
    n = args.atoms
    record = dict(atoms=n, elements=3 * n, past_int32=3 * n > 2**31 - 1, device=torch.cuda.get_device_name(),
                  torch=torch.__version__, checks={})
    started = time.perf_counter()
    positions64 = torch.empty((n, 3), dtype=torch.float64, device=device)
    fill(positions64)
    pos = positions64.to(torch.float32)
    record["fill_s"] = time.perf_counter() - started
    gen = torch.Generator(device="cpu").manual_seed(11)
    third = args.rows // 3
    # Slot rows at three places: the start, the middle, the end (the end's flat offsets 3i+k pass 2^31 from atom
    # 715,827,883 on); receivers first, as a slot's.
    nodes = torch.cat([torch.arange(0, third), torch.arange(n // 2, n // 2 + third),
                       torch.arange(n - (args.rows - 2 * third), n)]).to(device)
    receivers = args.rows // 2
    local = torch.randint(0, args.rows, (2, args.edges), generator=gen).to(device)
    local[1] = torch.sort(torch.randint(0, receivers, (args.edges,), generator=gen)).values.to(device)
    ei = nodes[local]
    shifts = torch.zeros((args.edges, 3), dtype=torch.float32, device=device)
    edge_mask = torch.ones(args.edges, dtype=torch.float32, device=device)
    edge_mask[-1000:] = 0
    node_mask = torch.ones(args.rows, dtype=torch.float32, device=device)
    far = torch.tensor([13.0, 0.0, 0.0], dtype=torch.float32, device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device) * 2300.0
    inverse = torch.linalg.inv(cell)
    small64 = positions64[nodes].clone()
    small32 = pos[nodes].clone()
    rows = torch.arange(args.rows, device=device)

    def check(name, big, small):
        same = bool(torch.equal(big, small))
        record["checks"][name] = dict(bitwise=same, largest_difference=float((big.double() - small.double()).abs().max()))
        print(name, record["checks"][name], flush=True)

    vectors = torch.compile(tiled._tile_vectors, dynamic=False)
    check("tile_vectors (float32 [N, 3], compiled)", vectors(pos, ei, shifts, edge_mask, far),
          vectors(small32, local, shifts, edge_mask, far))
    check("tile_vectors (eager)", tiled._tile_vectors(pos, ei, shifts, edge_mask, far),
          tiled._tile_vectors(small32, local, shifts, edge_mask, far))
    local_vectors = torch.compile(tiled._slot_local_vectors, dynamic=False)
    check("slot_local_vectors (float64 [N, 3], compiled)",
          local_vectors(positions64, nodes, node_mask, local, edge_mask, shifts, cell, inverse, receivers,
                        torch.float32, far),
          local_vectors(small64, rows, node_mask, local, edge_mask, shifts, cell, inverse, receivers,
                        torch.float32, far))
    # Small integers: every sum is exact in float32, so the atomics' order (index_add_ on CUDA) cannot change it and
    # only an index error can (Alex 4984768: random normals differed by 3e-5 at 1M atoms as at 750M, the order).
    values = torch.randint(-1024, 1025, (args.edges, 3), generator=gen).to(device, torch.float32)
    del pos
    forces = torch.zeros((n, 3), dtype=torch.float32, device=device)
    forces.index_add_(0, ei[1], values, alpha=-1)
    forces.index_add_(0, ei[0], values)
    small = torch.zeros((args.rows, 3), dtype=torch.float32, device=device)
    small.index_add_(0, local[1], values, alpha=-1)
    small.index_add_(0, local[0], values)
    check("forces index_add_ ([N, 3])", forces[nodes], small)
    touched = torch.zeros(n, dtype=torch.bool, device=device)
    touched[nodes] = True
    record["checks"]["forces untouched rows zero"] = dict(bitwise=bool(forces[~touched].abs().max() == 0))
    del touched
    largest = float(forces.abs().max())
    record["checks"]["forces abs max"] = dict(bitwise=largest == float(small.abs().max()))
    chunked = max(float(part.abs().max()) for part in positions64.split(1 << 26))
    whole = float(positions64.abs().max())
    record["checks"]["positions64 abs max"] = dict(bitwise=whole == chunked, value=whole)
    tail = forces.reshape(-1)[3 * (n - 10):]
    record["checks"]["flat view tail"] = dict(bitwise=bool(torch.equal(tail.reshape(10, 3), forces[n - 10:])))
    record["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    record["seconds"] = time.perf_counter() - started
    failed = [name for name, entry in record["checks"].items() if entry.get("bitwise") is False]
    record["failed"] = failed
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overflow-probe.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(failed=failed, peak_gib=record["peak_gib"], seconds=record["seconds"])))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
