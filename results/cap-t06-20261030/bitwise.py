"""CAP-T06: a reproducible float64 MH-1 trajectory with the capped noise batch equals the one before, bitwise.

One run per process: ``bitwise.py OUT.json new DIRECTORY STRUCTURE REPEAT STEPS SAVE_AT [CAP_STEPS]`` or
``bitwise.py OUT.json resume DIRECTORY STEPS [CAP_STEPS]``. ``CAP_STEPS`` sets
``fastmace.integrators.langevin.NOISE_BATCH_BYTES`` to that many Langevin steps of this cell (branch only), so a
small cell shows the batch shorter than a restart's unused rows. Writes the step, the positions' and momenta's
bytes (hex) and the noise batch's length.
"""

import json
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(fastmace.__file__).resolve().parents[1] / "tests"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase import units  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402

import fastmace as fm  # noqa: E402
from _support.structures import nbcl5  # noqa: E402

out, action, directory = sys.argv[1], sys.argv[2], sys.argv[3]
if action == "new":
    structure, repeat, steps, save_at = sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
    cap = int(sys.argv[8]) if len(sys.argv) > 8 else None
    natoms = len(nbcl5()) * int(np.prod([int(r) for r in repeat.split("x")]))
else:
    steps, save_at = int(sys.argv[4]), None
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else None
    natoms = None


def set_cap(natoms):
    if cap is not None:
        from fastmace.integrators import langevin
        langevin.NOISE_BATCH_BYTES = cap * 8 * 6 * natoms


started = time.perf_counter()
if action == "resume":
    natoms = len(np.load(Path(directory) / "restart.npz")["species"])
    set_cap(natoms)
    sim = fm.Simulation.resume(directory)
else:
    set_cap(natoms)
    assert structure == "NbCl5"
    atoms = nbcl5().repeat(tuple(int(r) for r in repeat.split("x")))
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(11))
    integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=300.0, friction_per_fs=0.02 * units.fs)
    ff = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64")
    sim = fm.Simulation(atoms, ff, integrator=integrator, seed=20261020,
                        workload=fm.Workload(max_temperature_K=300.0, reproducible=True))
if save_at:
    Path(directory).mkdir(parents=True, exist_ok=True)
    sim.run(steps=save_at)
    sim.save_restart(directory)
    sim.run(steps=steps - save_at)
else:
    sim.run(steps=steps)
committed = sim.committed()
batch = sim._engine.state.rng.batch
record = {"action": action, "natoms": natoms, "cap_steps": cap, "step": committed.step,
          "positions": committed.positions.tobytes().hex(), "momenta": committed.momenta.tobytes().hex(),
          "energy": committed.energy, "noise_batch": None if batch is None else int(batch.shape[0]),
          "fastmace": getattr(fastmace, "__version__", None), "seconds": time.perf_counter() - started,
          "gpu": torch.cuda.get_device_name()}
sim.close()
Path(out).write_text(json.dumps(record))
print(json.dumps({key: value for key, value in record.items() if key not in ("positions", "momenta")}))
