"""CAP-T08: does recompute_node_segments change whether the captured force graphs replay bitwise?

``replay.py OUT.json REPEAT STEPS``: in one process, for ``recompute_node_segments`` "off" and then "on", two
``fm.Simulation`` runs of MH-1 hybrid float32 on NbCl5 ``REPEAT`` with the same seed and thermalised start,
``STEPS`` steps each. Every step replays the captured force graphs; the second run captures its own. Writes, per
setting, whether the two runs' positions, momenta and energy agree bitwise. The default float32 plan has
non-deterministic records (atomic additions), so "off" gives the reference: the record must not make it worse.
"""

import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(fastmace.__file__).resolve().parents[1] / "tests"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402

import fastmace as fm  # noqa: E402
from _support.structures import nbcl5  # noqa: E402
from fastmace.config import RewriteConfig  # noqa: E402

out, repeat, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
loaded = fm.load("mh1-r2scan")
record = {"repeat": repeat, "steps": steps, "gpu": torch.cuda.get_device_name()}
for setting in ("off", "on"):
    runs = []
    for _ in range(2):
        atoms = nbcl5().repeat(tuple(int(r) for r in repeat.split("x")))
        MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(11))
        ff = fm.ForceField(loaded, atoms, precision="float32",
                           rewrites=RewriteConfig(recompute_node_segments=setting))
        sim = fm.Simulation(atoms, ff, seed=20261030, integrator=fm.integrators.Langevin(
            timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002))
        sim.run(steps=steps)
        committed = sim.committed()
        info = sim._engine.info()
        runs.append((committed.positions.copy(), committed.momenta.copy(), committed.energy,
                     sim.plan.rewrites.recompute_node_segments, info))
        sim.close()
    (p0, m0, e0, on0, info0), (p1, m1, e1, on1, _) = runs
    record[setting] = {
        "natoms": int(p0.shape[0]), "planned": [on0, on1], "positions_equal": bool(np.array_equal(p0, p1)),
        "momenta_equal": bool(np.array_equal(m0, m1)), "energy_equal": e0 == e1, "energy": [e0, e1],
        "max_position_difference": float(np.abs(p0 - p1).max()),
        "engine": {key: info0.get(key) for key in ("engine", "mode", "capture", "recaptures", "neighbour_rebuilds")}}
Path(out).write_text(json.dumps(record, indent=1, default=str))
print(json.dumps(record, default=str))
