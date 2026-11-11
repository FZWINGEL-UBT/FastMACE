"""Edge count and far-edge fraction (r >= R_FAR) at the model cutoff, per structure (nested radial c/d split)."""
import json
import sys

import numpy as np
from ase.neighborlist import neighbor_list

sys.path.insert(0, "tools")
import ab_step_time  # noqa: E402

cutoff, r_far = 6.0, 3.1518
rows = {}
for spec in sys.argv[2:]:
    name, atoms = ab_step_time.read_structure(spec, None)
    for jitter in (0.0, 0.07):
        work = atoms.copy()
        if jitter:
            work.positions += np.random.default_rng(0).normal(0.0, jitter, work.positions.shape)
        d = neighbor_list("d", work, cutoff)
        rows[f"{spec} jitter {jitter}"] = dict(atoms=len(work), edges=int(d.size),
                                               edges_per_atom=d.size / len(work),
                                               far_fraction=float((d >= r_far).mean()))
        print(spec, jitter, rows[f"{spec} jitter {jitter}"], flush=True)
json.dump(dict(cutoff=cutoff, r_far=r_far, rows=rows), open(sys.argv[1], "w"), indent=2)
