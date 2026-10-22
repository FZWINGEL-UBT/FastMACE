"""tools/bench/event_speed.py without a GPU: swap pairs, exchange momenta, the event side, the verdict."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("_tool_event_speed", TOOLS / "bench" / "event_speed.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load()


def test_swap_pair_takes_one_atom_of_each_kind_from_the_seeded_stream():
    species = np.array([11, 17, 11, 17, 11, 17, 17])
    first = [tool.swap_pair(species, (11, 17), np.random.default_rng(5)) for _ in range(3)]
    assert first[0] == first[1] == first[2]  # the same seed gives the same pair
    rng = np.random.default_rng(5)
    pairs = [tool.swap_pair(species, (11, 17), rng) for _ in range(40)]
    assert all(species[i] == 11 and species[j] == 17 for i, j in pairs)
    assert len(set(pairs)) > 3
    with pytest.raises(ValueError, match="no atom"):
        tool.swap_pair(np.array([11, 11]), (11, 17), rng)


def test_exchange_swaps_species_and_momenta_and_keeps_kinetic_energy_and_total_momentum():
    species = np.array([11, 17, 17])
    masses = np.array([22.99, 35.45, 35.45])
    momenta = np.arange(9, dtype=float).reshape(3, 3)
    new_species, new_momenta = tool.exchange(species, momenta, 0, 2)
    assert new_species.tolist() == [17, 17, 11]
    assert new_momenta[0].tolist() == momenta[2].tolist() and new_momenta[2].tolist() == momenta[0].tolist()
    assert np.allclose(new_momenta.sum(0), momenta.sum(0))
    new_masses = masses[[2, 1, 0]]  # the masses follow the species
    kinetic = lambda p, m: float((p ** 2 / m[:, None]).sum())  # noqa: E731
    assert kinetic(new_momenta, new_masses) == pytest.approx(kinetic(momenta, masses))
    assert species.tolist() == [11, 17, 17]  # inputs untouched


class FakeSimulation:
    def __init__(self, species):
        self.species = np.array(species)
        self.momenta = np.arange(3 * len(species), dtype=float).reshape(-1, 3)
        self.calls = []
        self.step = 0

    def run(self, steps):
        self.calls.append(("run", steps))
        self.step += steps

    def committed(self):
        return SimpleNamespace(step=self.step, species=self.species.copy(), momenta=self.momenta.copy())

    def update(self, *, species=None, momenta=None):
        self.calls.append(("update", species.tolist(), momenta.tolist()))
        self.species, self.momenta = np.asarray(species), np.asarray(momenta)


def test_the_event_side_runs_a_chunk_then_swaps_and_the_fixed_side_only_runs():
    sim = FakeSimulation([11, 17, 11, 17])
    side = tool.SimulationSide(sim, every=100, kinds=(11, 17), seed=3)
    side.step()
    side.step()
    assert [call[0] for call in sim.calls] == ["run", "update", "run", "update"]
    assert sim.calls[0] == ("run", 100) and side.events == 2
    assert sorted(sim.species.tolist()) == [11, 11, 17, 17]  # composition kept
    fixed_sim = FakeSimulation([11, 17])
    fixed = tool.SimulationSide(fixed_sim, every=100, kinds=None, seed=3)
    fixed.step()
    assert fixed_sim.calls == [("run", 100)] and fixed.events == 0


def test_swap_back_undoes_every_other_event_so_the_composition_returns():
    start = [11, 17, 11, 17, 11, 17]
    sim = FakeSimulation(start)
    side = tool.SimulationSide(sim, every=100, kinds=(11, 17), seed=3, swap_back=True)
    side.step()
    first = [call for call in sim.calls if call[0] == "update"][0]
    assert first[1] != start
    side.step()
    assert sim.species.tolist() == start  # the second event swapped the same two atoms back
    updates = [call for call in sim.calls if call[0] == "update"]
    assert np.allclose(updates[1][2], np.arange(18, dtype=float).reshape(-1, 3))  # momenta back too
    side.step()
    assert sim.species.tolist() != start and side.events == 3


def test_the_side_records_the_edge_count_before_each_block():
    counts = iter([7000, 7100, 7200])
    side = tool.SimulationSide(FakeSimulation([11, 17]), every=100, kinds=None, seed=0, edges=lambda: next(counts))
    side.prepare_block()
    side.prepare_block()
    assert side.block_edges == [7000, 7100]


def test_entries_after_takes_the_log_entries_added_while_timed():
    log = [dict(step=10, key="A"), dict(step=20, key="('B', 8192)", reason="new edge size")]
    assert tool.entries_after(log, 1) == [log[1]]
    assert tool.entries_after(log, 2) == []


def test_steps_must_be_whole_event_intervals():
    common = ["--output", "o", "--checkpoint", "c"]
    args = tool.parse_args(common)
    assert (args.temperature_K, args.timestep_fs, args.every, args.warmup, args.block, args.blocks) == (
        1500.0, 1.0, 100, 300, 300, 12)
    assert tool.chunk_protocol(args).as_dict() == dict(warmup=3, block=3, blocks=12)
    for bad in (["--block", "250"], ["--warmup", "150"]):
        with pytest.raises(SystemExit):
            tool.parse_args(common + bad)


def test_summary_reports_per_step_times_the_rate_ratio_and_the_verdict():
    medians = {"fixed": 300.0, "events": 320.0}  # ms per 100-step chunk
    summary = tool.summarise(medians, every=100, threshold=0.9)
    assert summary["ms_per_step"] == {"fixed": 3.0, "events": 3.2}
    assert summary["rate_ratio"] == pytest.approx(300 / 320)
    assert summary["pass"] is True
    assert tool.summarise({"fixed": 300.0, "events": 340.0}, every=100, threshold=0.9)["pass"] is False


class CountingSide:
    def __init__(self, captures):
        self.captures = list(captures)  # captures logged after each step
        self.steps = 0

    def step(self):
        self.steps += 1


def test_settle_warms_up_until_enough_chunks_pass_without_a_new_capture():
    a, b = CountingSide([]), CountingSide([])
    logs = {"a": [1, 2, 2, 2, 2, 2, 2], "b": [0, 0, 1, 1, 1, 1, 1]}

    def captures():
        return {"a": logs["a"][a.steps - 1] if a.steps else 0, "b": logs["b"][b.steps - 1] if b.steps else 0}

    chunks = tool.settle({"a": a, "b": b}, captures, quiet=3, limit=10)
    assert chunks == 6 and a.steps == b.steps == 6  # b's capture in chunk 3, then 3 quiet chunks
    assert tool.settle({"a": a, "b": b}, captures, quiet=0, limit=10) == 0
    noisy = {"a": CountingSide([]), "b": CountingSide([])}
    count = iter(range(100))
    assert tool.settle(noisy, lambda: {"a": next(count), "b": 0}, quiet=2, limit=4) == 4  # capped


def test_settle_arguments():
    args = tool.parse_args(["--output", "o", "--checkpoint", "c", "--settle", "3", "--settle-max", "2000"])
    assert (args.settle, args.settle_max, args.swap_back) == (3, 2000, False)
    assert tool.parse_args(["--output", "o", "--checkpoint", "c", "--swap-back"]).swap_back is True
    with pytest.raises(SystemExit):
        tool.parse_args(["--output", "o", "--checkpoint", "c", "--settle-max", "250"])


def test_a_recapture_or_compile_while_timed_invalidates_the_run():
    before = {"fixed": 2, "events": 5}
    assert tool.validity(before, {"fixed": 2, "events": 5}, recompiled=False, others=[]) == (True, [])
    valid, why = tool.validity(before, {"fixed": 2, "events": 6}, recompiled=False, others=[])
    assert not valid and "events" in why[0]
    assert not tool.validity(before, before, recompiled=True, others=[])[0]
    assert not tool.validity(before, before, recompiled=False, others=[1234])[0]
