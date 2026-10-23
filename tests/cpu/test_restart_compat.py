"""A restart from before phase 4 resumes without ``allow_plan_change`` (4-T13; SPEC §10.6: older files load).

Phase 4 added integrator plan fields (``ensemble``, ``scheme``, ``dof``, ``parameters``; 4-T01). A format-3 plan
record has none of them. Read back at their defaults, ``dof`` came out ``"auto"`` while a fresh plan records the
resolved ``"fastmace"``, so ``Simulation.resume`` refused an unchanged run with "the plan differs from the
restart's in integrator" (seen with a restart of the 3b freeze, ff80ca2). A field that a format-3 record does not
have is read at what that run did; a field that differs still refuses.

The format-3 files are made the way ``test_contracts_4.py::test_format_3_files_load_without_fixed_atoms`` makes
them: a format-4 restart with the phase 4 entries taken out and the version set to 3.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_simulation import _positions, _simulation, cpu_planner, loaded  # noqa: F401  (fixtures)

from fastmace.errors import RestartMismatch
from fastmace.io import restart as restart_io
from fastmace.protocols import hold

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

PHASE_4_INTEGRATOR_FIELDS = ("ensemble", "scheme", "dof", "parameters")


def _as_format_3(path, **plan_integrator):
    """Rewrite the restart at ``path`` as fastmace-3 wrote it: no phase 4 plan, scheme or fixed-atom entries."""
    stored = dict(restart_io.read_restart(path))
    plan = json.loads(str(stored["plan"]))
    for name in PHASE_4_INTEGRATOR_FIELDS:
        if name != "ensemble":  # fastmace-3 records had the ensemble
            plan["integrator"].pop(name, None)
    plan["integrator"].update(plan_integrator)
    physical = json.loads(str(stored["physical"]))
    physical.pop("fixed_atoms", None)
    integrator = json.loads(str(stored["integrator"]))
    integrator["config"].pop("scheme", None)
    stored.update(plan=np.asarray(json.dumps(plan)), physical=np.asarray(json.dumps(physical)),
                  integrator=np.asarray(json.dumps(integrator)), fastmace_restart=np.asarray(3))
    with open(path, "wb") as handle:
        np.savez(handle, **stored)
    assert restart_io.format_version(restart_io.read_restart(path)) == 3


def _saved(loaded, tmp_path, **plan_integrator):  # noqa: F811
    sim = _simulation(loaded, seed=2)
    sim.run(schedule=hold(300.0, ps=0.005), steps=2)
    sim.save_restart(tmp_path / "run" / restart_io.RESTART_NAME)
    _as_format_3(tmp_path / "run" / restart_io.RESTART_NAME, **plan_integrator)
    return sim


def test_a_format_3_restart_resumes_without_allow_plan_change_and_continues_bitwise(loaded, tmp_path):  # noqa: F811
    from fastmace.simulation import Simulation

    sim = _saved(loaded, tmp_path)
    resumed = Simulation.resume(tmp_path / "run")
    assert resumed.plan.revision == sim.plan.revision  # nothing changed: no new revision
    assert "allow_plan_change" not in resumed.plan.reason
    sim.run(steps=3)
    resumed.run(steps=3)
    assert np.array_equal(_positions(resumed), _positions(sim))


@pytest.mark.parametrize("field, value", [("dof", "ase"), ("scheme", "baoab"), ("ensemble", "npt")])
def test_a_field_the_restart_has_and_that_differs_still_refuses(loaded, tmp_path, field, value):  # noqa: F811
    from fastmace.simulation import Simulation

    _saved(loaded, tmp_path, **{field: value})
    with pytest.raises(RestartMismatch, match="integrator"):
        Simulation.resume(tmp_path / "run")


def test_a_real_integrator_change_in_a_format_3_restart_still_refuses(loaded, tmp_path):  # noqa: F811
    from fastmace.simulation import Simulation

    sim = _saved(loaded, tmp_path)
    plan = json.loads(str(restart_io.read_restart(tmp_path / "run" / restart_io.RESTART_NAME)["plan"]))
    _as_format_3(tmp_path / "run" / restart_io.RESTART_NAME, friction=2 * plan["integrator"]["friction"])
    with pytest.raises(RestartMismatch, match="integrator"):
        Simulation.resume(tmp_path / "run")
    resumed = Simulation.resume(tmp_path / "run", allow_plan_change=True)
    assert resumed.plan.revision == sim.plan.revision + 1
