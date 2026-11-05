"""SrTiO3 10x10x10 (5,000 atoms) in three atom orders for the kernel lab's x-gather locality test (round 4):
ASE repeat order (the benchmark's), Morton (Z-order of the fractional coordinates, 10 bits per axis) and a seeded
random shuffle (the falsifier: no locality). Writes extxyz files next to this script."""
from pathlib import Path

import numpy as np
from ase.io import read, write

here = Path(__file__).resolve().parent
atoms = read(here.parents[2] / "results/loup-a100-20261103/SrTiO3.cif").repeat((10, 10, 10))
atoms.set_pbc(True)


def morton(frac, bits=10):
    q = np.clip((np.mod(frac, 1.0) * (1 << bits)).astype(np.int64), 0, (1 << bits) - 1)
    code = np.zeros(len(q), dtype=np.int64)
    for b in range(bits):
        for d in range(3):
            code |= ((q[:, d] >> b) & 1) << (3 * b + d)
    return code


orders = {"repeat": np.arange(len(atoms)),
          "morton": np.argsort(morton(atoms.get_scaled_positions()), kind="stable"),
          "random": np.random.default_rng(0).permutation(len(atoms))}
for name, order in orders.items():
    write(here / f"SrTiO3-10x10x10-{name}.extxyz", atoms[order])
    print(name, len(order))
