"""OPT6-T02 smoke: the default plan against chunked_conv_backward=on, one rattled cell each, in one process.

Prints the plan decision, the chunk size of the GPU (l2_chunk_edges for MH-1's Q = 1280, float32) and the
largest energy and force differences. The streaming backward's atomic sender sums differ run to run, so the
forces agree to rounding, not bitwise. Usage: python smoke.py NaCl NbCl5 NaNbO3
"""

import json
import sys
from pathlib import Path

import fastmace as fm  # before torch: soft file locks

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fastmace.config import RewriteConfig  # noqa: E402
from fastmace.kernels.oeq_streaming import l2_chunk_edges  # noqa: E402

model = fm.load("mh1-r2scan")
for spec in sys.argv[1:]:
    name, atoms = ab_step_time.read_structure(spec, None)
    atoms.positions += np.random.default_rng(20261027).uniform(-0.05, 0.05, atoms.positions.shape)
    out = {}
    for side, rewrites in (("default", RewriteConfig()), ("chunked", RewriteConfig(chunked_conv_backward="on")),
                           ("default2", RewriteConfig())):
        ff = fm.ForceField(model, atoms, precision="float32", rewrites=rewrites)
        decision = list(ff.evaluation_plan(atoms).decision("RewriteConfig.chunked_conv_backward"))
        ff.evaluate(atoms)
        energy, forces, _ = ff.evaluate(atoms)
        out[side] = dict(energy=float(energy), forces=np.asarray(forces, dtype=np.float64), decision=decision)
        del ff
        torch.cuda.empty_cache()
    d, c, d2 = out["default"], out["chunked"], out["default2"]
    print(json.dumps(dict(structure=str(name), natoms=len(atoms), decision=c["decision"][:2],
                          default_decision=d["decision"][:2], chunk_edges=l2_chunk_edges("cuda", 1280, 4),
                          energy_diff_eV=c["energy"] - d["energy"],
                          max_force_diff=float(np.abs(c["forces"] - d["forces"]).max()),
                          default_repeat_max_force_diff=float(np.abs(d2["forces"] - d["forces"]).max()),
                          max_force=float(np.abs(d["forces"]).max()))), flush=True)
