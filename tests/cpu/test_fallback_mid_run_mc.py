"""Mid-run fallbacks and the MC move set (plan task 5-T08b; SPEC §9.4 step 5, §7.6, §10.6; decision 54).

A fallback goes back to the last committed canonical point. The move set attached there comes back from the
point's record; a move set attached, replaced or detached after the point (a later ``run(mc=...)``) is redone
at its step, with the streams, counters and cycle position it had then. So the MC random numbers, the
accepted moves and the counters of the redone steps are an uninterrupted run's:
- against the unbroken run, the MC record (streams, counters, cycle position) and the species are equal, and
  the trajectory agrees to float64 rounding (on the CPU the compiled tier steps as the graph tier does);
- ``counters.mc``'s ranges cover the steps run with a move set attached once.

The failure is a capture error tagged as ``GraphEngine._captured`` tags it, raised by one step (monkeypatched
here only). MC blocks run on the CPU through the tiny MH-1 model's real driver.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_adapt_monitor as monitor_test

from fastmace import integrators
from fastmace.config import AdaptationConfig, EngineConfig
from fastmace.engine import graph
from fastmace.planner import fallback
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning"), pytest.mark.rule(14)]

quiet_licences = monitor_test.quiet_licences
pristine = monitor_test.pristine

CAPTURE_ERROR = "operation not permitted when stream is capturing (cudaErrorStreamCaptureUnsupported, injected)"
WORKLOAD = dict(max_temperature_K=600.0, species_changes=True, min_mc_temperature_K=1.0, mc_moves=("swap",))


def swap(every=3, attempts=2):
    from fastmace.mc import Swap

    return Swap(pairs=(("Na", "Cl"),), every=every, attempts=attempts)


def simulation(pristine, output=None):
    from _support.structures import nacl
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.02)
    return Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=integrator, seed=5,
                      workload=Workload(**WORKLOAD), output=output, engine=EngineConfig(device="cpu",
                                                                                        compile_mode="none"),
                      adaptation=AdaptationConfig(adapt="off"))


@pytest.fixture
def break_step(monkeypatch):
    """``arm(step)``: the step to ``step`` raises a tagged capture error, once; ``hits`` lists the steps."""
    original = graph.GraphEngine.step
    state = {"at": None, "hits": []}

    def step(self):
        if state["at"] is not None and int(self.state.step) + 1 == state["at"]:
            state["at"] = None
            state["hits"].append(int(self.state.step) + 1)
            error = RuntimeError(CAPTURE_ERROR)
            error.fastmace_stage = graph.CAPTURE_STAGE
            raise error
        return original(self)

    monkeypatch.setattr(graph.GraphEngine, "step", step)

    def arm(at):
        state["at"] = at

    arm.hits = state["hits"]
    return arm


def assert_same_run(broken, reference):
    first, second = broken.committed(), reference.committed()
    assert first.step == second.step
    np.testing.assert_array_equal(first.species, second.species)  # the same accepted swaps
    np.testing.assert_allclose(first.positions, second.positions, rtol=0, atol=1e-9)
    np.testing.assert_allclose(first.momenta, second.momenta, rtol=0, atol=1e-9)


def mc_state(sim):
    return sim._mc.state_dict() if sim._mc is not None else sim._mc_record


def test_a_move_set_attached_after_the_point_is_redone_with_its_streams(pristine, break_step):
    """The point is the setup's (no restart write); the move set is attached at step 10; the failure at 25."""
    def run(sim):
        sim.run(steps=10)
        sim.run(steps=30, mc=swap())

    reference = simulation(pristine)
    run(reference)
    broken = simulation(pristine)
    break_step(25)
    run(broken)
    assert break_step.hits == [25] and fallback.is_mid_run(broken.plan) and broken.plan.step == 0
    assert reference.plan.tier == "graph" and broken.plan.tier != "graph"
    assert mc_state(broken) == mc_state(reference)  # streams, counters, cycle position: the unbroken run's
    assert mc_state(broken)["counters"]["swap"]["attempted"] > 0
    assert broken._mc_ranges == reference._mc_ranges == [[10, 40]]
    assert_same_run(broken, reference)
    broken.close()
    reference.close()


def test_a_detach_after_the_point_is_redone(pristine, break_step, tmp_path):
    """The point (a restart write at step 10) has the move set attached; run(mc=None) detaches it at 10."""
    from fastmace.io.output import Output

    def run(sim):
        sim.run(steps=10, mc=swap())
        sim.run(steps=20, mc=None)

    reference = simulation(pristine, Output(tmp_path / "reference", trajectory_every=None, restart_every=10,
                                            thermo_every=5))
    run(reference)
    broken = simulation(pristine, Output(tmp_path / "broken", trajectory_every=None, restart_every=10,
                                         thermo_every=5))
    break_step(14)
    run(broken)
    assert break_step.hits == [14] and broken.plan.step == 10
    assert broken._mc is None and reference._mc is None
    assert broken._mc_record == reference._mc_record
    assert broken._mc_ranges == reference._mc_ranges == [[0, 10]]
    assert_same_run(broken, reference)
    broken.close()
    reference.close()


def test_a_replaced_move_set_comes_back_from_the_point_then_the_new_one(pristine, break_step, tmp_path):
    """Attached at the point (restart at 10), replaced at 10 by another move set; the failure at 18."""
    from fastmace.io.output import Output

    def run(sim):
        sim.run(steps=10, mc=swap())
        sim.run(steps=15, mc=swap(every=2, attempts=1))

    reference = simulation(pristine, Output(tmp_path / "reference", trajectory_every=None, restart_every=10,
                                            thermo_every=5))
    run(reference)
    broken = simulation(pristine, Output(tmp_path / "broken", trajectory_every=None, restart_every=10,
                                         thermo_every=5))
    break_step(18)
    run(broken)
    assert break_step.hits == [18] and broken.plan.step == 10
    assert mc_state(broken) == mc_state(reference)
    assert broken._mc_ranges == reference._mc_ranges == [[0, 25]]
    assert_same_run(broken, reference)
    broken.close()
    reference.close()
