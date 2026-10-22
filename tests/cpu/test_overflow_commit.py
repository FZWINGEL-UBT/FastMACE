"""Overflow before commit and the re-capture policy (plan task 3-T04, SPEC §8.1, §12.2 "replay and transitions").

A step is committed only after the host has checked every flag its evaluation can set: the edge count
against the padded edges the force stage used (pipelined and speculative steps), and, for PolarMACE
with a variable cell, the k-vector overflow flag after every force stage (one host sync, conductor
answer 1). An overflowed step restores what stage A produced, grows what overflowed and redoes the
force stage with the same noise (answer 2). The same restore-and-redo is exposed for MC attempts
(``GraphEngine.redo_on_overflow``), with the caller's random-number cursors restored.

The overflows are forced here, never by a hook in the library: the next padded size of a pipelined
or speculative step is shrunk below the coming count (a monkeypatched capacity), and the polar flag
is set inside one force call.

Re-captures: every capture is logged; one that replaces or adds a variant of a key already captured
(a new edge size, a capture after ``invalidate_graphs``, an eviction or a pool reset) counts against
``EngineConfig.recapture_budget``. Past the budget the engine warns, records it and recommends the
compiled tier, and keeps running. At most ``graph_variant_cap`` graphs are kept; the least recently
replayed is evicted (answers 3 and 4).
"""

import warnings

import numpy as np
import pytest
import torch
from test_canonical_restart import lj_engine, mace_builder, restart_atoms, stored  # noqa: F401 (fixture)
from test_engine import build

from _support.structures import nacl
from _support.tiny_models import polar_available
from fastmace import provenance
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.engine import graph
from fastmace.engine.graph import EDGE_OVERFLOW, KVECTOR_OVERFLOW, CapacityOverflow, RecaptureBudgetWarning
from fastmace.io import restart

STEPS = 40
FORCED = (5, 12, 19, 26, 33)  # validated steps after which the next force stage gets too few edges
SHRINK = 20  # edges below the last count; all within the cutoff (compacted to the front)


def heat(engine, steps):
    for step in range(1, steps + 1):
        engine.integrator.set_temperature(min(3.0 * step, 150.0))
        engine.step()


def force_edge_overflows(engine, monkeypatch, steps=FORCED):
    """Shrink the next force stage's edges after the validation of each step in ``steps``."""
    if engine.speculative:
        name, attribute = "_validate_oldest_speculative", "_spec_edges"
    else:
        name, attribute = "_validate_pipelined", "_pipe_edges"
    original = getattr(engine, name)
    forced = []

    def validate():
        original()
        if engine.state.step in steps and engine.state.step not in forced:
            forced.append(engine.state.step)
            setattr(engine, attribute, int(engine.force_field.edge_count) - SHRINK)

    monkeypatch.setattr(engine, name, validate)
    return forced


def truncated_calls(engine, monkeypatch):
    """Record the energy of every force call that read fewer edges than the compacted count."""
    field = engine.force_field
    original = type(field).__call__
    energies = []

    def call(self, positions):
        out = original(self, positions)
        if self is field and self.num_edges < int(self.nl.count):
            energies.append(float(out.energy[0]))
        return out

    monkeypatch.setattr(type(field), "__call__", call)
    return energies


# --------------------------------------------------------------------------- edge overflow in MD steps


@pytest.mark.parametrize("mode", ["pipelined", "speculative"])
def test_an_edge_overflow_is_redone_and_never_observed(tmp_path, monkeypatch, mode):
    """Every frame, thermo value and the restart come from validated evaluations, equal to checked steps."""
    reference = lj_engine(mode="checked")
    engine = lj_engine(mode=mode)
    forced = force_edge_overflows(engine, monkeypatch)
    truncated = truncated_calls(engine, monkeypatch)
    frames, expected = [], []
    for step in range(1, STEPS + 1):
        for each in (reference, engine):
            each.integrator.set_temperature(min(3.0 * step, 150.0))
            each.step()
        frames.append(engine.snapshot().wait())  # a frame and a thermo row (validates the step first)
        expected.append(reference.snapshot().wait())
        assert not engine.uncommitted()
    assert forced == list(FORCED)
    assert len(truncated) == len(FORCED)  # each forced step evaluated once on too few edges ...
    info = engine.info()
    assert info["overflows_redone"] == len(FORCED)  # ... and was redone
    assert [entry["step"] for entry in info["overflow_log"]] == [step + 1 for step in FORCED]
    assert all(entry["kinds"] == ["edges"] for entry in info["overflow_log"])
    for got, want in zip(frames, expected, strict=True):
        for key in want:
            np.testing.assert_allclose(got[key], want[key], rtol=0, atol=1e-10, err_msg=key)
    energies = [float(np.asarray(frame["energy"]).ravel()[0]) for frame in expected]
    for value in truncated:  # no frame carries a truncated evaluation's energy
        assert min(abs(value - energy) for energy in energies) > 1e-6
    file = stored(engine, tmp_path)  # a restart written right after the run: committed state only
    want = stored(reference, tmp_path, "reference.npz")
    assert "positions" in want and "velocities" in want
    for key in ("positions", "momenta", "velocities", "energy", "forces"):
        if key in want:
            np.testing.assert_allclose(file[key], want[key], rtol=0, atol=1e-10, err_msg=key)


def test_a_pipelined_step_in_flight_is_uncommitted_until_validated():
    engine = lj_engine(mode="pipelined")
    assert not engine.uncommitted()
    engine.step()
    assert engine.uncommitted()  # T08's restart write guard refuses here
    engine.flush()
    assert not engine.uncommitted()


def test_an_overflow_with_a_rebuild_counts_once_as_a_rebuild_redo(monkeypatch):
    """A rebuild redo is a stale list, not an overflow; only count overflows are counted as overflows."""
    engine = lj_engine(mode="pipelined", neighbours=NeighbourConfig(skin=0.3, sort_receivers="off",
                                                                     gpu_neighbour_list="off"))
    heat(engine, STEPS)
    engine.flush()
    info = engine.info()
    assert info["overflows_redone"] == info["pipelined_redos"] - info["pipelined_rebuild_redos"]


# --------------------------------------------------------------------------- polar k-vector overflow


def inject_polar_overflow(engine, monkeypatch, step):
    """Set the sticky polar flag inside the force stage that step ``step`` launches after its own stage A,
    never inside a redo (which runs before the next stage A, and which a real k-vector overflow cannot
    reach). Returns the force stages run, as (state.step, key)."""
    original = engine._run
    calls, armed = [], []

    def run(key):
        original(key)
        if key in ("AC", "ACP"):
            armed[:] = [engine.state.step == step - 1]
        elif isinstance(key, tuple) and key[0] == "B":
            calls.append(engine.state.step)
            if armed and armed[0]:
                armed[:] = []
                engine.force_field._polar_state().overflow.fill_(True)

    monkeypatch.setattr(engine, "_run", run)
    return calls


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("pipelined", ["off", "on"])
def test_a_kvector_overflow_is_caught_in_its_own_step(mace_builder, monkeypatch, pipelined):  # noqa: F811
    if not polar_available():
        pytest.skip("PolarMACE needs MACE's polar module")
    reference = mace_builder("polar", nacl(rattle=0.05, seed=1), npt=True, pipelined=pipelined)
    engine = mace_builder("polar", nacl(rattle=0.05, seed=1), npt=True, pipelined=pipelined)
    assert engine.info()["overflow_check"] == "per step"
    plan = mace_builder.plans[-1]
    assert any(item == "overflow check" for item, _, _ in plan.decisions)
    calls = inject_polar_overflow(engine, monkeypatch, step=4)
    for _ in range(4):
        reference.step()
        engine.step()
    engine.flush()
    reference.flush()
    assert not bool(engine.force_field._polar_state().overflow)  # cleared by the step's own check, no canonical point
    info = engine.info()
    assert info["canonical_points"] == 0
    assert info["polar_overflow_steps"] == [4]
    kvectors = [entry for entry in info["overflow_log"] if "k-vectors" in entry["kinds"]]
    assert [(entry["step"], entry["mode"]) for entry in kvectors] == [(4, engine.mode)]
    assert info["overflows_redone"] == len(info["overflow_log"])  # pipelined: edge-count redos too
    assert len(calls) >= 5  # four force stages and the redo of step 4
    got, want = engine.sync(), reference.sync()
    for key in want:
        np.testing.assert_allclose(got[key], want[key], rtol=1e-9, atol=1e-10, err_msg=key)


def test_a_fixed_cell_plan_has_no_per_step_check():
    engine = lj_engine(mode="checked")
    assert engine.info()["overflow_check"] == "edges"


# --------------------------------------------------------------------------- MC attempts


def test_redo_on_overflow_restores_and_repeats_with_the_same_numbers():
    engine = lj_engine(mode="checked")
    start = engine.state.positions.clone()
    offset = torch.zeros((), dtype=torch.int64)  # stands in for an MC stream's device offset
    numbers = []

    def attempt():
        generator = torch.Generator().manual_seed(1234 + int(offset))
        draw = torch.rand(3, generator=generator, dtype=torch.float64)
        numbers.append(draw)
        engine.state.positions[0] += 0.01 * draw
        offset.add_(1)
        if len(numbers) == 1:  # the first evaluation overflows its edges
            engine.flag_edge_overflow(torch.tensor(10), torch.tensor(5))

    redos = engine.redo_on_overflow(attempt, keep=(offset,))
    assert redos == 1
    torch.testing.assert_close(numbers[0], numbers[1], rtol=0, atol=0)
    assert int(offset) == 1
    expected = start.clone()
    expected[0] += 0.01 * numbers[0]
    torch.testing.assert_close(engine.state.positions, expected, rtol=0, atol=0)
    info = engine.info()
    assert info["overflows_redone"] == 1
    assert info["overflow_log"][-1]["kinds"] == ["edges"]
    assert not engine.overflowed()


def test_redo_on_overflow_gives_up_with_the_internal_signal():
    engine = lj_engine(mode="checked")

    def attempt():
        engine.flag_edge_overflow(torch.tensor(10), torch.tensor(5))

    with pytest.raises(CapacityOverflow) as raised:
        engine.redo_on_overflow(attempt, max_redos=2)
    assert raised.value.word == EDGE_OVERFLOW
    assert EDGE_OVERFLOW != KVECTOR_OVERFLOW


def test_flag_edge_overflow_is_a_device_operation():
    engine = lj_engine(mode="checked")
    engine.flag_edge_overflow(torch.tensor(5), torch.tensor(5))
    assert engine.overflowed() == 0
    engine.flag_edge_overflow(torch.tensor(6), torch.tensor(5))
    assert engine.overflowed() == EDGE_OVERFLOW
    engine.canonicalise()  # a restart point clears it (registry: reset, sticky)
    assert engine.overflowed() == 0


# --------------------------------------------------------------------------- the re-capture policy


def replay_engine(**engine):
    """A checked replay engine whose padded edge size cycles through three values (the argon cell's count
    stays put at these temperatures): every step can need a new B variant. Padding is exact."""
    config = EngineConfig(pipelined="off", async_output="off", compile_mode="none", device="cpu", **engine)
    out = build(engine=config, replay=True)
    capacity = int(out.force_field.edge_capacity)
    out._padded = lambda count: min(capacity, count + out.state.step % 3)
    return out


def test_first_captures_are_logged_and_new_variants_counted():
    engine = replay_engine()
    engine.prepare()
    log = engine.info()["recapture_log"]
    assert [entry["key"] for entry in log] == ["AC", f"('B', {engine.initial_edges})"]
    assert [entry["counted"] for entry in log] == [False, False]
    assert {entry["reason"] for entry in log} == {"first capture"}
    heat(engine, STEPS)  # every step may need a new B size
    info = engine.info()
    counted = [entry for entry in info["recapture_log"] if entry["counted"]]
    assert info["recaptures"] == len(counted) > 0
    assert all(entry["reason"] == "new variant of B" for entry in counted)
    first = {entry["key"] for entry in info["recapture_log"] if not entry["counted"]}
    assert first - {"C"} == {"AC", f"('B', {engine.initial_edges})"}  # C: at the first rebuild, if any


def test_invalidate_graphs_drops_them_and_the_next_capture_counts():
    engine = replay_engine()
    engine.prepare()
    before = engine.info()["recaptures"]
    engine.invalidate_graphs("friction changed")
    assert not engine._graphs
    engine.step()
    info = engine.info()
    assert info["invalidations"][-1]["reason"] == "friction changed"
    reasons = [entry["reason"] for entry in info["recapture_log"] if entry["counted"]]
    assert reasons.count("friction changed") >= 2  # AC and B
    assert info["recaptures"] >= before + 2


def test_invalidate_graphs_drops_only_the_keys_given():
    engine = replay_engine()
    engine.prepare()
    engine.invalidate_graphs("test", keys=["AC"])
    assert list(engine._graphs) == [("B", engine._padded(engine.initial_edges))]


@pytest.mark.filterwarnings("ignore::fastmace.engine.graph.RecaptureBudgetWarning")
def test_the_variant_cap_evicts_the_least_recently_used_graph():
    engine = replay_engine(graph_variant_cap=2)
    held = []
    for step in range(1, STEPS + 1):
        engine.integrator.set_temperature(min(3.0 * step, 150.0))
        engine.step()
        held.append(len(engine._graphs))
    assert max(held) <= 2
    info = engine.info()
    assert info["graph_evictions"] > 0
    evicted = [entry for entry in info["recapture_log"] if entry["reason"] == "evicted by the variant cap"]
    assert evicted and all(entry["counted"] for entry in evicted)


def test_the_budget_warns_records_and_recommends_compiled_without_downgrading():
    engine = replay_engine(recapture_budget=0)
    engine.prepare()
    with pytest.warns(RecaptureBudgetWarning, match="compiled"):
        heat(engine, STEPS)
    info = engine.info()
    assert info["recapture_budget"] == 0
    assert info["recapture_budget_exceeded_at"] is not None
    assert "compiled" in engine.recapture_warning
    assert info["mode"] == "checked" and engine.capture  # no mid-run downgrade
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engine.step()  # warned once


def test_an_auto_budget_without_a_plan_is_the_variant_cap():
    assert replay_engine().info()["recapture_budget"] == 64
    assert replay_engine(graph_variant_cap=8).info()["recapture_budget"] == 8


def test_captured_wraps_other_modules_graphs():
    engine = replay_engine()
    runs = []
    replay = engine.captured("mc.attempt", lambda: runs.append(1))
    assert runs == []  # captured on its first replay
    replay()
    replay()
    assert len(runs) == 2
    assert "mc.attempt" in engine._graphs
    engine.invalidate_graphs("test")
    replay()
    entries = [entry for entry in engine.info()["recapture_log"] if entry["key"] == "mc.attempt"]
    assert [entry["counted"] for entry in entries] == [False, True]
    for key in ("AC", "E", ("B", 4), ("ACB", 4, 0)):
        with pytest.raises(ValueError, match="engine"):
            engine.captured(key, lambda: None)


def test_captured_runs_eagerly_without_capture():
    engine = lj_engine(mode="checked")
    runs = []
    engine.captured("mc.attempt", lambda: runs.append(1))()
    assert runs == [1]
    assert not engine._graphs


def test_a_pool_reset_marks_the_dropped_graphs_for_a_counted_recapture():
    engine = replay_engine()
    engine.prepare()
    engine.invalidate_graphs(graph.POOL_RESET)
    engine.step()
    assert graph.POOL_RESET in {entry["reason"] for entry in engine.info()["recapture_log"] if entry["counted"]}


# --------------------------------------------------------------------------- provenance counters


def test_provenance_counters_come_from_the_engine():
    engine = replay_engine()
    heat(engine, 10)
    info = engine.info()
    record = provenance.metadata_record(engine_info=info)
    assert record["counters"]["recaptures"] == info["recaptures"] > 0
    assert record["counters"]["overflows_redone"] == info["overflows_redone"] == 0
    given = provenance.metadata_record(engine_info=info, counters={"recaptures": 7})
    assert given["counters"]["recaptures"] == 7  # an explicit value wins
    assert provenance.metadata_record()["counters"]["recaptures"] is None


def test_the_auto_budget_is_resolved_in_the_plan():
    from fastmace.planner.fixed import FixedPlanner

    planner = FixedPlanner(engine=EngineConfig(device="cpu", graph_variant_cap=16), environ={})
    budget = planner._phase_3_fields(_Book(), "float64", polar=False, npt=False, evaluation=False)
    assert budget["engine"]["recapture_budget"] == 16


class _Book:
    def __init__(self):
        self.decisions = []

    def decide(self, item, choice, reason):
        self.decisions.append((item, str(choice), reason))


def test_resume_after_an_overflow_equals_the_uninterrupted_run(tmp_path, monkeypatch):
    """A restart written after a redone overflow resumes bit for bit (SPEC §12.2 restart row)."""
    whole = lj_engine(mode="pipelined")
    force_edge_overflows(whole, monkeypatch, steps=(9,))
    heat(whole, 10)
    file = stored(whole, tmp_path)
    assert whole.info()["overflows_redone"] == 1
    fresh = lj_engine(restart_atoms(__import__("test_engine").argon(), file), mode="pipelined")
    restart.resume_engine(file, fresh)
    for engine in (whole, fresh):
        for step in range(11, 21):
            engine.integrator.set_temperature(min(3.0 * step, 150.0))
            engine.step()
    got, want = fresh.sync(), whole.sync()
    for key in want:
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)
