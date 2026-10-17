"""Validation and A/B tools (plan task 1a-T18): step-time statistics, alternating A/B blocks, the baseline
loader and the force/stress gate rule, all on synthetic numbers.

Nothing here builds a model or touches a GPU. ``tools/force_stress_gate.py`` is loaded from its file;
its research-engine and GPU code runs only in its ``main``.
"""

import importlib.util
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from fastmace.bench import ab, step_time

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", REPOSITORY_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_tool("force_stress_gate")


class FakeClock:
    """A clock that advances only when a fake engine steps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeEngine:
    """Records every call; each step advances the clock by the next duration in ``step_seconds``."""

    def __init__(self, name, clock, step_seconds, log):
        self.name, self.clock, self.log = name, clock, log
        self.step_seconds = list(step_seconds)
        self.steps = 0

    def step(self):
        self.clock.now += self.step_seconds[self.steps % len(self.step_seconds)]
        self.steps += 1
        self.log.append((self.name, "step"))

    def prepare_block(self):
        self.log.append((self.name, "prepare"))

    def finish_block(self):
        self.log.append((self.name, "finish"))

    def synchronize(self):
        self.log.append((self.name, "sync"))


# --- timing protocol and statistics (SPEC §12.5) -------------------------------------------------------


def test_protocol_defaults_are_the_spec_protocol():
    protocol = step_time.TimingProtocol()
    assert (protocol.warmup, protocol.block, protocol.blocks) == (300, 300, 4)


@pytest.mark.parametrize("fields", [dict(block=0), dict(blocks=0), dict(warmup=-1)])
def test_protocol_rejects_empty_blocks(fields):
    with pytest.raises(ValueError):
        step_time.TimingProtocol(**fields)


def test_block_statistics():
    stats = step_time.BlockStatistics.from_blocks([4.0, 3.0, 5.0, 3.5])
    assert stats.median_ms == pytest.approx(3.75)
    assert stats.mean_ms == pytest.approx(3.875)
    assert (stats.min_ms, stats.max_ms) == (3.0, 5.0)
    assert stats.stdev_ms == pytest.approx(statistics.stdev([4.0, 3.0, 5.0, 3.5]))
    assert stats.spread == pytest.approx(2.0 / 3.75)
    record = stats.as_dict()
    assert record["block_ms"] == [4.0, 3.0, 5.0, 3.5]
    assert record["median_ms"] == pytest.approx(3.75)
    json.dumps(record)


def test_block_statistics_of_one_block():
    stats = step_time.BlockStatistics.from_blocks([2.0])
    assert stats.median_ms == 2.0 and stats.stdev_ms == 0.0 and stats.spread == 0.0


def test_block_statistics_rejects_no_blocks():
    with pytest.raises(ValueError):
        step_time.BlockStatistics.from_blocks([])


def test_time_engine_warms_up_then_times_each_block():
    clock, log = FakeClock(), []
    # 10 warm-up steps at 1 s (not timed), then blocks of 5 steps at 2, 3, 4 ms.
    durations = [1.0] * 10 + [0.002] * 5 + [0.003] * 5 + [0.004] * 5
    engine = FakeEngine("a", clock, durations, log)
    stats = step_time.time_engine(engine, step_time.TimingProtocol(warmup=10, block=5, blocks=3), clock=clock)
    assert stats.block_ms == pytest.approx((2.0, 3.0, 4.0))
    assert stats.median_ms == pytest.approx(3.0)
    assert engine.steps == 25
    calls = [call for _, call in log]
    # Warm-up ends with a flush and a synchronisation before the first timed block.
    assert calls[10:12] == ["finish", "sync"]
    # Every block: prepare, synchronise, steps, flush, synchronise.
    assert calls[12:21] == ["prepare", "sync"] + ["step"] * 5 + ["finish", "sync"]


def test_result_record_is_versioned_json():
    stats = step_time.BlockStatistics.from_blocks([1.0, 2.0])
    record = step_time.result_record(stats, setup_s=12.5, peak_memory_bytes=1 << 30)
    assert record["schema_version"] == step_time.SCHEMA_VERSION
    assert record["setup_s"] == 12.5 and record["peak_memory_bytes"] == 1 << 30
    assert record["median_ms"] == pytest.approx(1.5)
    json.dumps(record)


# --- alternating A/B blocks ---------------------------------------------------------------------------


def test_block_order_alternates():
    assert ab.block_order(("a", "b"), 3) == ["a", "b", "a", "b", "a", "b"]


def test_block_order_abba():
    assert ab.block_order(("a", "b"), 4, pattern="abba") == ["a", "b", "b", "a", "a", "b", "b", "a"]


def test_block_order_gives_every_side_the_same_number_of_blocks():
    order = ab.block_order(("a", "b"), 5, pattern="abba")
    assert order.count("a") == order.count("b") == 5


@pytest.mark.parametrize("names, blocks, pattern", [(("a",), 2, "abab"), (("a", "a"), 2, "abab"),
                                                     (("a", "b"), 0, "abab"), (("a", "b"), 2, "xyz")])
def test_block_order_rejects_bad_arguments(names, blocks, pattern):
    with pytest.raises(ValueError):
        ab.block_order(names, blocks, pattern=pattern)


def test_run_ab_warms_up_both_then_alternates_blocks():
    clock, log = FakeClock(), []
    sides = {"a": FakeEngine("a", clock, [0.002], log), "b": FakeEngine("b", clock, [0.003], log)}
    result = ab.run_ab(sides, step_time.TimingProtocol(warmup=4, block=2, blocks=3), clock=clock)
    # Both warm-ups come before any timed block.
    first_prepare = next(i for i, (_, call) in enumerate(log) if call == "prepare")
    assert [name for name, call in log[:first_prepare] if call == "step"] == ["a"] * 4 + ["b"] * 4
    # Timed blocks then alternate a, b, a, b, a, b.
    assert [name for name, call in log if call == "prepare"] == ["a", "b"] * 3
    assert result.order == ("a", "b") * 3
    assert result.statistics["a"].median_ms == pytest.approx(2.0)
    assert result.statistics["b"].median_ms == pytest.approx(3.0)
    assert result.ratio == pytest.approx(1.5)
    assert result.pair_ratios == pytest.approx((1.5, 1.5, 1.5))
    record = result.as_dict()
    assert record["order"] == ["a", "b"] * 3 and record["ratio_b_over_a"] == pytest.approx(1.5)
    json.dumps(record)


def test_run_ab_pairs_blocks_in_order():
    clock, log = FakeClock(), []
    # b is 1 % slower in the first pair and 3 % slower in the second.
    sides = {"a": FakeEngine("a", clock, [0.0] * 1 + [0.001, 0.001], log),
             "b": FakeEngine("b", clock, [0.0] * 1 + [0.00101, 0.00103], log)}
    result = ab.run_ab(sides, step_time.TimingProtocol(warmup=1, block=1, blocks=2), clock=clock)
    assert result.pair_ratios == pytest.approx((1.01, 1.03))


def test_run_ab_needs_two_sides():
    clock, log = FakeClock(), []
    with pytest.raises(ValueError):
        ab.run_ab({"a": FakeEngine("a", clock, [0.001], log)}, step_time.TimingProtocol(1, 1, 1), clock=clock)


# --- the baseline loader ------------------------------------------------------------------------------


def _export(root, commit="d2faa9fdca47d7f6f55e59a0d871ce0af2e53ffd"):
    package = root / "Experiments" / "Langevin_Thermostat" / "langevin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (root / "Experiments" / "Benchmark_MACE_GPU_OPTIMIZATIONS" / "scripts").mkdir(parents=True)
    if commit is not None:
        (root / "COMMIT").write_text(commit + "\n")
    return root


def test_loader_reads_the_recorded_commit(tmp_path):
    export = ab.load_baseline(_export(tmp_path))
    assert export.commit == "d2faa9fdca47d7f6f55e59a0d871ce0af2e53ffd"
    assert export.root == tmp_path.resolve()
    assert export.engine_root == tmp_path.resolve() / "Experiments" / "Langevin_Thermostat"
    assert export.as_dict()["commit"] == export.commit


def test_loader_accepts_a_short_commit(tmp_path):
    assert ab.load_baseline(_export(tmp_path, commit="d2faa9fd")).commit == "d2faa9fd"


def test_loader_refuses_an_export_without_a_commit_file(tmp_path):
    with pytest.raises(ab.BaselineExportError, match="COMMIT"):
        ab.load_baseline(_export(tmp_path, commit=None))


@pytest.mark.parametrize("commit", ["", "research-engine-baseline-20261017", "d2fa", "xyz12345"])
def test_loader_refuses_a_commit_file_without_a_commit(tmp_path, commit):
    with pytest.raises(ab.BaselineExportError, match="COMMIT"):
        ab.load_baseline(_export(tmp_path, commit=commit))


def test_loader_refuses_a_directory_without_the_engine(tmp_path):
    (tmp_path / "COMMIT").write_text("d2faa9fd\n")
    with pytest.raises(ab.BaselineExportError, match="langevin"):
        ab.load_baseline(tmp_path)


def test_loader_refuses_a_missing_directory(tmp_path):
    with pytest.raises(ab.BaselineExportError):
        ab.load_baseline(tmp_path / "missing")


def test_activate_puts_the_export_first_once(tmp_path):
    export = ab.load_baseline(_export(tmp_path))
    path = ["/somewhere"]
    ab.activate(export, path)
    ab.activate(export, path)
    assert path[:2] == [str(export.engine_root), str(export.benchmark_scripts)]
    assert len(path) == 3


def test_check_origin_refuses_a_module_from_elsewhere(tmp_path):
    export = ab.load_baseline(_export(tmp_path / "export"))
    inside = export.engine_root / "langevin" / "__init__.py"
    ab.check_origin(export, "langevin", str(inside))
    with pytest.raises(ab.BaselineExportError, match="outside"):
        ab.check_origin(export, "langevin", str(tmp_path / "other" / "langevin" / "__init__.py"))


# --- the force/stress gate (SPEC §12.2, §12.3) ----------------------------------------------------------


def test_errors():
    reference = np.array([[1.0, -4.0, 0.0], [2.0, 0.0, 0.5]])
    candidate = reference + np.array([[0.0, 0.2, 0.0], [-0.1, 0.0, 0.0]])
    errors = gate.errors(reference, candidate)
    assert errors["max_abs"] == pytest.approx(0.2)
    assert errors["rms"] == pytest.approx(np.sqrt((0.04 + 0.01) / 6))
    assert errors["reference_max_abs"] == 4.0


def test_floors_are_the_spec_floors():
    assert gate.GATE_RATIO == 2.0
    assert gate.FLOORS == {"forces": 1e-5, "stress": 1e-7}


@pytest.mark.parametrize("production, yardstick, passed", [
    (2.0e-4, 1.0e-4, True),     # exactly twice the float32 error
    (2.01e-4, 1.0e-4, False),   # just above
    (0.5e-4, 1.0e-4, True),     # better than float32
])
def test_ratio_rule(production, yardstick, passed):
    assert gate.quantity_verdict("forces", production, yardstick)["passed"] is passed


def test_force_floor_where_float32_is_nearly_exact():
    # float32 happens to be nearly exact: 2 x 1e-7 would fail 8e-6, the floor of 1e-5 eV/A passes it.
    verdict = gate.quantity_verdict("forces", 8e-6, 1e-7)
    assert verdict["passed"] and verdict["limit"] == 1e-5
    assert not gate.quantity_verdict("forces", 1.1e-5, 1e-7)["passed"]


def test_stress_floor():
    assert gate.quantity_verdict("stress", 9e-8, 1e-9)["passed"]
    assert not gate.quantity_verdict("stress", 1.1e-7, 1e-9)["passed"]
    assert gate.quantity_verdict("stress", 1.1e-7, 1e-9)["limit"] == 1e-7


def test_zero_yardstick_uses_the_floor():
    verdict = gate.quantity_verdict("forces", 5e-6, 0.0)
    assert verdict["passed"] and verdict["ratio"] is None


def _config(production_forces, f32_forces, production_stress, f32_stress, e_prod=0.001, e_f32=0.002):
    return dict(production=dict(forces=dict(max_abs=production_forces), stress=dict(max_abs=production_stress),
                                energy_meV_atom=e_prod),
                ase_f32=dict(forces=dict(max_abs=f32_forces), stress=dict(max_abs=f32_stress),
                             energy_meV_atom=e_f32))


def test_summary_takes_the_worst_configuration_of_each_path():
    configs = [_config(3e-4, 2e-4, 1e-6, 2e-6), _config(1e-4, 1e-4, 5e-6, 1e-6)]
    summary = gate.summarize(configs)
    assert summary["forces"]["production_max_abs"] == 3e-4
    assert summary["forces"]["ase_f32_max_abs"] == 2e-4
    assert summary["forces"]["ratio"] == pytest.approx(1.5) and summary["forces"]["passed"]
    assert summary["stress"]["production_max_abs"] == 5e-6
    assert summary["stress"]["ratio"] == pytest.approx(2.5) and not summary["stress"]["passed"]
    assert summary["energy_meV_atom"] == dict(production=0.001, ase_f32=0.002)
    assert summary["passed"] is False


def test_summary_passes_when_both_quantities_pass():
    assert gate.summarize([_config(1e-4, 1e-4, 1e-6, 1e-6)])["passed"] is True


def test_random_strain_is_symmetric_and_bounded(rng):
    hydrostatic = gate.random_strain(rng, 0.02, full=False)
    assert np.allclose(hydrostatic, hydrostatic[0, 0] * np.identity(3))
    assert abs(hydrostatic[0, 0] - 1) <= 0.02
    full = gate.random_strain(rng, 0.02, full=True)
    assert np.allclose(full, full.T)
    assert np.abs(full - np.identity(3)).max() <= 0.02
