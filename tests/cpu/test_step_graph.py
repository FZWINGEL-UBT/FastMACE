"""``EngineConfig.step_graph`` (opt-2 OPT2-T04) on the CPU: one ("ACPB", edges) stage per pipelined step.

ACP, the flags' copy to the host, the event the next validation waits on, and B run as one stage, so
on CUDA one graph replay per step instead of two. The kernels and their order are those of ACP and B,
so the trajectory is bitwise the default's. Checked here eagerly and with test_prepare_edges' recorder
in place of the capture (padded edges; the recorder replays the edge views active at capture, as a
CUDA graph does); the 2-T21d guard covers the new key.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_prepare_edges as edges_test

from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.engine import graph
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

# test_prepare_edges' fixtures: the tiny MH-1-family model and the recorder in place of the capture.
quiet_licences = edges_test.quiet_licences
pristine = edges_test.pristine
recorded_capture = edges_test.recorded_capture


def simulation(pristine, **engine):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = edges_test.nacl(repeat=1, rattle=0.0, seed=3)
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)  # the edge count grows within 50 steps (2-T21d)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    return Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=integrator, seed=8,
                      workload=Workload(max_temperature_K=300.0),
                      engine=EngineConfig(device="cpu", compile_mode="none", pipelined="on", **engine))


def run(pristine, steps=100, **engine):
    sim = simulation(pristine, **engine)
    sim.run(steps=steps)
    out = sim.committed(), sorted(str(key) for key in sim._engine._graphs), sim._engine.pipelined_redos
    sim.close()
    return out


def test_the_option_is_off_by_default_and_carried_by_the_plan(pristine):
    assert EngineConfig().step_graph is False
    sim = simulation(pristine, step_graph=True)
    assert sim.plan.engine.step_graph is True and sim._engine.pipelined
    sim.close()


def test_eager_step_graph_equals_the_default_bitwise(pristine):
    (default, _, redos), (merged, _, redos_merged) = run(pristine), run(pristine, step_graph=True)
    assert redos == redos_merged
    np.testing.assert_array_equal(default.positions, merged.positions)
    np.testing.assert_array_equal(default.momenta, merged.momenta)
    np.testing.assert_array_equal(default.forces, merged.forces)


def test_captured_step_graph_reads_its_keys_edges_and_equals_the_default(pristine, recorded_capture, monkeypatch):
    """One ACPB capture per edge size, each reading its key's edges; bitwise the uncaptured default."""
    merged, graphs, _ = run(pristine, step_graph=True)
    acpb = [(key, count) for key, count in recorded_capture if isinstance(key, tuple) and key[0] == "ACPB"]
    assert acpb and all(count == key[1] for key, count in acpb), recorded_capture
    assert not any(key == "ACP" for key, _ in recorded_capture)  # no separate ACP replay
    monkeypatch.undo()
    edges_test.pad_edges(monkeypatch)
    default, _, _ = run(pristine)
    np.testing.assert_array_equal(default.positions, merged.positions)
    np.testing.assert_array_equal(default.momenta, merged.momenta)
    np.testing.assert_array_equal(default.forces, merged.forces)


def test_the_capture_guard_covers_the_new_key(pristine, recorded_capture):
    sim = simulation(pristine, step_graph=True)
    engine = sim._engine
    key = ("ACPB", engine._padded(engine.initial_edges))
    assert key[1] != engine.initial_edges
    engine._graphs.pop(key, None)
    engine.force_field.use_edges(engine.initial_edges)
    with pytest.raises(RuntimeError, match="active edges"):
        engine._captured(key)
    sim.close()
    assert graph.GraphEngine._key_edges(key) == key[1]
