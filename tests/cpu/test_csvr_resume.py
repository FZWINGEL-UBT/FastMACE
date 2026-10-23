"""``Simulation.resume`` of a CSVR run (4-T13's preflight, 23 October 2026).

CSVR refuses to start at rest, as ASE's ``Bussi`` does (``integrators/csvr.py``). A resume used to build its
engine from the restart's positions and cell with the momenta still zero, and to load the restart's state only
afterwards, so every CSVR resume failed with "Initial kinetic energy is zero" (GPU, reproducible mode, float64 and
float32: ``tests/gpu/test_reproducible_schemes_gpu.py[csvr-*]``). The rebuilt ``Atoms`` now carry the restart's
momenta. A fresh run at rest is still refused.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from test_simulation import _positions, _simulation, cpu_planner, loaded  # noqa: F401  (fixtures)

from _support.structures import nacl
from fastmace import integrators
from fastmace.io import restart as restart_io

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

CSVR = integrators.CSVR(timestep_fs=1.0, temperature_K=300.0, tau_fs=20.0)


def _thermal():
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(7))
    Stationary(atoms)
    return atoms


def test_a_csvr_run_resumes_and_continues_bitwise(loaded, tmp_path):  # noqa: F811
    from fastmace.simulation import Simulation

    sim = _simulation(loaded, atoms=_thermal(), integrator=CSVR, seed=5)
    sim.run(steps=3)
    sim.save_restart(tmp_path / "run" / restart_io.RESTART_NAME)
    resumed = Simulation.resume(tmp_path / "run")
    sim.run(steps=4)  # across a fresh noise batch row and the batch's cursor in the restart (SPEC §10.6)
    resumed.run(steps=4)
    assert np.array_equal(_positions(resumed), _positions(sim))
    assert np.array_equal(resumed.committed().momenta, sim.committed().momenta)


def test_a_fresh_csvr_run_at_rest_is_still_refused(loaded):  # noqa: F811
    atoms = nacl(repeat=1, rattle=0.05, seed=3)  # no momenta
    with pytest.raises(ValueError, match="kinetic energy is zero"):
        _simulation(loaded, atoms=atoms, integrator=CSVR, seed=5)
