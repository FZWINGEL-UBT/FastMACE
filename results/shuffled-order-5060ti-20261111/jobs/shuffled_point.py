"""B4 (11 November 2026): MD step time of SrTiO3 with its atoms in ASE's repeat order against a random permutation.

tools/bench/capacity_scan.py ``point`` (one engine per process, the planner's default plan, OMAT-0 medium), with
two changes made here: with SHUFFLE=1 the supercell's atoms are permuted (numpy default_rng(SHUFFLE_SEED)) right
after ``Atoms.repeat``, before the velocities are drawn; and the process's CUDA memory fraction is capped at
MEMORY_FRACTION (0.9) right after the engine configures its allocator. Usage: the capacity_scan point arguments.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ["FM_SRC"]) / "tools" / "bench"))
import capacity_scan as cs  # noqa: E402

SHUFFLE = os.environ.get("SHUFFLE") == "1"
original_base_cell = cs.base_cell


def base_cell(structure):
    name, cell = original_base_cell(structure)
    base = type(cell)

    class Shuffled(base):
        def repeat(self, rep):
            atoms = base.repeat(self, rep)
            atoms.__class__ = base
            order = np.random.default_rng(int(os.environ.get("SHUFFLE_SEED", "0"))).permutation(len(atoms))
            return atoms[order]

    if SHUFFLE:
        cell.__class__ = Shuffled
        name += "-shuffled"
    return name, cell


cs.base_cell = base_cell

import fastmace.model.compile as compile_  # noqa: E402

original_configure = compile_.configure_allocator


def configure_allocator(*args, **kwargs):
    result = original_configure(*args, **kwargs)
    import torch

    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMORY_FRACTION", "0.9")))
    return result


compile_.configure_allocator = configure_allocator

if __name__ == "__main__":
    sys.argv[0] = str(Path(cs.__file__))
    sys.exit(cs.main())
