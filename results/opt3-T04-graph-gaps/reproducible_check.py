"""OPT3-T04: combo_kernels in the reproducible mode (2-T16): bitwise the default, across fresh processes.

    python reproducible_check.py OUT.npz off|on

One fresh process per call (``reproducible=True`` must come before CUDA starts). MH-1 plain cuEq float32 on
NaCl 3×3×3, the reproducible plan, 200 NVT steps from a fixed seed; positions, momenta and forces are saved.
Run off, off again and on; the two "off" runs are bitwise equal (the mode's qualification), and so must "on" be.
"""

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import os
import sys

import numpy as np
from ase.io import read

import fastmace as fm
from fastmace.config import RewriteConfig

out, combo = sys.argv[1], sys.argv[2]
atoms = read(os.path.join(os.environ["FASTMACE_STRUCTURES"], "Structures/Benchmark/R2SCAN/NaCl.cif")).repeat((3, 3, 3))
atoms.set_pbc(True)
atoms.rattle(0.02, seed=5)
model = fm.load("mh1-r2scan")
ff = fm.ForceField(model, atoms, precision="float32", rewrites=RewriteConfig(combo_kernels=combo))
integrator = fm.integrators.Langevin(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.01)
sim = fm.Simulation(atoms, ff, integrator=integrator, seed=11,
                    workload=fm.Workload(max_temperature_K=600.0, reproducible=True))
plan = sim.plan
assert plan.decision("RewriteConfig.combo_kernels")[1] == combo, plan.decision("RewriteConfig.combo_kernels")
sim.run(steps=200)
committed = sim.committed()
np.savez(out, positions=committed.positions, momenta=committed.momenta, forces=committed.forces,
         energy=committed.energy)
print(f"{combo}: backend {plan.engine.backend}, conv fusion {plan.engine.cueq_conv_fusion}, "
      f"energy {committed.energy!r}")
sim.close()
