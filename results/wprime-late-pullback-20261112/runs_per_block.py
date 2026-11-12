"""STAGE_DOUT runs per 32-edge block for SrTiO3 10x10x10 (OMAT-0 medium cutoff 6 A), edges grouped by receiver in
atom order (the plan's receiver-sorted edge list). Perfect perovskite, a = 3.905 A (the MD snapshot differs slightly)."""
import json
import numpy as np
from ase.build import bulk
from ase.neighborlist import neighbor_list
from common import HERE

atoms = bulk("SrTiO3", "cubic", a=3.905) if False else None
from ase.spacegroup import crystal
atoms = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                cellpar=[3.905, 3.905, 3.905, 90, 90, 90]).repeat((10, 10, 10))
i, j = neighbor_list("ij", atoms, 6.0)
recv = np.sort(i)
E = len(recv)
blocks = [len(np.unique(recv[s:s + 32])) for s in range(0, E, 32)]
res = dict(atoms=len(atoms), edges=int(E), edges_per_atom=E / len(atoms), runs_per_block=float(np.mean(blocks)))
print(res)
(HERE / "runs_per_block.json").write_text(json.dumps(res, indent=1) + "\n")
