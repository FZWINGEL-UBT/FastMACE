"""``metadata.json`` of a real run carries the phase 2 keys (plan task 2-T17; SPEC §13).

A tiny MH-1-like model on the CPU, as in ``test_simulation.py``: the per-record self-checks of the
built plan, the always-on treatments, the strain path the force field bound (2-T08b), the runner,
and no subprocess block for a plan that runs in this process.
"""

import fastmace  # noqa: F401  (before torch: it switches PyTorch's file locks, SPEC §14)

# isort: split
import json

import pytest
from test_simulation import _simulation, cpu_planner, loaded  # noqa: F401  (fixtures)

from fastmace.io.output import Output

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]


def test_a_run_records_the_phase_2_keys(loaded, tmp_path):  # noqa: F811
    directory = tmp_path / "run"
    sim = _simulation(loaded, output=Output(directory, thermo_every=1, trajectory_every=None, restart_every=None))
    sim.run(steps=1)
    sim.close()
    metadata = json.loads((directory / "metadata.json").read_text())
    plan = sim._engine.plan  # the built plan: MinimalPlanner.build adds the self-checks (2-T04)
    assert plan.revision == sim.plan.revision
    assert metadata["plan"]["runner"] == "in-process" and metadata["subprocess"] is None
    assert metadata["plan"]["fallbacks_tried"] == []
    assert [entry["name"] for entry in metadata["gate"]["selfchecks"]] == [check.name for check in plan.selfchecks]
    assert metadata["gate"]["selfchecks"], "the built plan applied rewrites with self-checks"
    assert {entry["name"] for entry in metadata["software"]["treatments"]} >= {"float64_energy", "single_graph_strain"}
    bound = sim._engine.force_field.strain_path
    assert metadata["force_field"]["strain_path"] == bound and bound in ("one_graph", "mace")
    assert metadata["plan"]["engine"]["dynamic_edges"] in (True, False)
    assert metadata["plan"]["workload"]["reproducible"] is False
