"""``tools/bench/ase_integrators_speed.py`` (4-T13): the class maps and the timing, without a GPU."""

import importlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tool():
    path = ROOT / "tools" / "bench" / "ase_integrators_speed.py"
    spec = importlib.util.spec_from_file_location("_ase_integrators_speed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_class_exists_in_ase_and_in_the_shim():
    import fastmace.ase

    tool = _tool()
    frontends = tool._tool("ab_frontends")
    assert set(tool.STOCK_CLASSES) == set(frontends.INTEGRATORS)
    for name, module in tool.STOCK_CLASSES.items():
        assert hasattr(importlib.import_module(module), name), (module, name)
        assert hasattr(fastmace.ase, name), name
        frontends.ase_arguments(name, 300.0)  # every class has its arguments


def test_measure_times_one_block_after_the_warm_up():
    tool = _tool()
    calls, ticks = [], iter([0.0, 2.0, 2.0, 2.6])

    class Dynamics:
        def run(self, steps):
            calls.append(steps)

    warmup_s, ms = tool.measure(Dynamics(), lambda: calls.append("sync"), 300, 200, clock=lambda: next(ticks))
    assert calls == [300, "sync", 200, "sync"]
    assert warmup_s == 2.0 and ms == pytest.approx(3.0)


def test_the_arguments():
    tool = _tool()
    args = tool.parse_args(["--side", "stock", "--integrator", "MTKNPT", "--output", "out"])
    assert (args.structure, args.timestep_fs, args.warmup, args.block) == ("NaCl", 1.0, 300, 300)
    with pytest.raises(SystemExit):
        tool.parse_args(["--side", "ase", "--integrator", "MTKNPT", "--output", "out"])


def test_nve_drift_fits_the_slope_and_the_scatter_about_it():
    path = ROOT / "tools" / "bench" / "nve_drift.py"
    spec = importlib.util.spec_from_file_location("_nve_drift", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    times = [0.0, 1.0, 2.0, 3.0]
    slope, rms, spread = tool.drift(times, [1.0 + 0.5 * t + (0.1 if i % 2 else -0.1) for i, t in enumerate(times)])
    assert slope == pytest.approx(0.5 + 0.04, abs=1e-12)  # the alternating scatter tilts the fit a little
    assert 0.0 < rms < 0.1 and spread == pytest.approx(1.7)
    assert tool.parse_args(["--output", "o"]).ps == 20.0
