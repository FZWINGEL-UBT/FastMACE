"""tools/validation/npt_d3_comparison.py (plan 3b-T05): the ASE side's run loop and the comparison's statistics.

The statistics on synthetic AR(1) series with known answers: the integrated autocorrelation time, block
standard errors pooled over seeds, the jackknife of the standard deviation, and the 3-SE rule passing equal
engines and failing a shifted one. The run loop on Lennard-Jones argon: one row per sample, the pressure
equal to ASE's stress with the ideal-gas part.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones

TOOL = Path(__file__).resolve().parents[2] / "tools" / "validation" / "npt_d3_comparison.py"


def _tool():
    spec = importlib.util.spec_from_file_location("_tool_npt_d3_comparison", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _tool()


def ar1(n, phi, sigma, mean, rng):
    """An AR(1) series with stationary standard deviation ``sigma``; tau = (1 + phi) / (1 - phi)."""
    noise = rng.normal(0.0, sigma * math.sqrt(1 - phi * phi), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return mean + x


def test_integrated_time_of_ar1():
    rng = np.random.default_rng(1)
    taus = [tool.integrated_time(ar1(20000, 0.8, 1.0, 0.0, rng)) for _ in range(4)]
    assert np.mean(taus) == pytest.approx(9.0, rel=0.2)  # (1 + 0.8) / (1 - 0.8)
    assert tool.integrated_time(rng.normal(size=5000)) == pytest.approx(1.0, abs=0.3)


def test_block_standard_error_matches_the_ar1_value():
    rng = np.random.default_rng(2)
    phi, sigma, n = 0.8, 0.4, 3000
    tau = (1 + phi) / (1 - phi)
    errors, estimates = [], []
    for _ in range(40):
        series = [ar1(n, phi, sigma, 29.0, rng) for _ in range(3)]
        blocks = [block for s in series for block in tool.blocks_of(s, math.ceil(2 * tau) * 5)]
        mean, se = tool.block_mean(blocks)
        errors.append(mean - 29.0)
        estimates.append(se)
    true_se = sigma * math.sqrt(tau / (3 * n))
    assert np.std(errors) == pytest.approx(true_se, rel=0.35)
    assert np.mean(estimates) == pytest.approx(true_se, rel=0.35)


def test_jackknife_of_the_standard_deviation():
    rng = np.random.default_rng(3)
    blocks = tool.blocks_of(rng.normal(0.0, 2.0, 40000), 1000)
    std, se = tool.jackknife_std(blocks)
    assert std == pytest.approx(2.0, rel=0.02)
    assert se == pytest.approx(2.0 / math.sqrt(2 * 40000), rel=0.4)  # iid normal: sigma / sqrt(2N)


def synthetic_runs(rng, shift=0.0, temperature=300.0, n=2000, equilibration_ps=0.0):
    runs = []
    for engine, offset, seeds in (("ase", 0.0, (101, 102, 103)), ("fastmace", shift, (1, 2, 3))):
        for seed in seeds:
            time_ps = np.arange(1, n + 1) * 0.01 + equilibration_ps
            data = dict(time_ps=time_ps, temperature_K=np.full(n, temperature),
                        volume_A3_atom=ar1(n, 0.9, 0.4, 29.0 + offset, rng),
                        pressure_GPa=ar1(n, 0.5, 0.3, 0.0, rng), potential_eV_atom=ar1(n, 0.7, 0.002, -4.5, rng))
            runs.append(({"engine": engine, "seed": seed, "temperature_K": temperature,
                          "equilibration_ps": equilibration_ps}, data))
    return runs


def test_equal_engines_pass_and_a_shifted_volume_fails():
    rng = np.random.default_rng(4)
    passed = tool.analyse(synthetic_runs(rng))["300K"]
    assert passed["passed"], passed["comparison"]
    assert passed["block_samples"] >= 2 * passed["tau_volume_samples"]
    assert passed["engines"]["ase"]["seeds"] == [101, 102, 103]
    se = passed["comparison"]["volume_A3_atom"]["combined_se"]
    shifted = tool.analyse(synthetic_runs(rng, shift=6 * se))["300K"]
    assert not shifted["passed"]
    assert not shifted["comparison"]["volume_A3_atom"]["passed"]
    assert shifted["comparison"]["volume_A3_atom"]["difference"] > 0  # fastmace minus ase


def test_one_engine_only_is_reported_not_passed():
    rng = np.random.default_rng(5)
    runs = [run for run in synthetic_runs(rng) if run[0]["engine"] == "ase"]
    entry = tool.analyse(runs)["300K"]
    assert "passed" not in entry and set(entry["engines"]) == {"ase"}


def test_read_run_drops_the_equilibration(tmp_path):
    directory = tmp_path / "ase-300K-s101"
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps({"engine": "ase", "seed": 101, "temperature_K": 300.0,
                                                    "equilibration_ps": 0.05}))
    rows = [tool.COLUMNS] + [(step, step / 1000, 300, -4.5, 0.04, 29.0, 0.0, 19.0, 18.0, 18.0)
                             for step in range(0, 101, 10)]
    (directory / "samples.csv").write_text("\n".join(",".join(str(v) for v in row) for row in rows) + "\n")
    record, data = tool.read_run(directory)
    assert record["seed"] == 101
    np.testing.assert_allclose(data["time_ps"], [0.06, 0.07, 0.08, 0.09, 0.1])


def test_run_loop_samples_and_pressure_with_the_kinetic_part():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    calculator = LennardJones(sigma=3.4, epsilon=0.0104, rc=7.0, smooth=True)
    rows = []

    def check(row):
        rows.append(row)
        stress = atoms.get_stress(voigt=False, include_ideal_gas=True)
        assert row[tool.COLUMNS.index("pressure_GPa")] == pytest.approx(-np.trace(stress) / 3 / tool_units_GPa(),
                                                                         rel=1e-9, abs=1e-12)

    final = tool.run_ase(atoms, calculator, 50.0, 7, 30, 10, check)
    assert [row[0] for row in rows] == [0, 10, 20, 30]
    assert rows[-1][tool.COLUMNS.index("volume_A3_atom")] == pytest.approx(final.get_volume() / len(final))
    assert len(rows[0]) == len(tool.COLUMNS)


def tool_units_GPa():
    from ase import units

    return units.GPa


def test_settings_are_the_agreed_ones():
    settings = tool.SETTINGS
    assert settings["repeat"] == (1, 1, 3) and settings["timestep_fs"] == 1.0
    assert settings["barostat_tau_fs"] == 2000.0 and settings["barostat_mass_factor"] == 1.0
    assert settings["friction_per_ase_time"] == 0.02 and settings["pressure_GPa"] == 0.0
    assert tool.SIGMAS == 3.0 and tool.MIN_BLOCK_TAU == 2.0


def test_d3_record_is_the_r2scan_bj_set():
    record = tool.d3_record()
    assert (record.s6, record.s8, record.a1, record.a2) == (1.0, 0.7898, 0.4948, 5.7308)  # conductor's choice 2
    assert record.damping == "bj" and not record.three_body and record.pair_cutoff_bohr == 40.0


def test_fastmace_side_runs_a_simulation_with_d3(tmp_path, monkeypatch):
    """``fastmace_side`` on a tiny model (CPU planner): one row per sample, the cell moves, the plan has D3."""
    pytest.importorskip("cuequivariance_torch")
    import torch

    from _support.mc_fixtures import cpu_planner
    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model

    cpu_planner(monkeypatch)
    path = tmp_path / "tiny-mh1-11-17.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    settings = {**tool.SETTINGS, "head": None}
    rows = []
    atoms = nacl(1, rattle=0.05, seed=3)
    simulation = tool.fastmace_side(path, atoms, 300.0, 5, 20, 5, rows.append, settings=settings)
    assert [row[0] for row in rows] == [0, 5, 10, 15, 20]
    volumes = [row[tool.COLUMNS.index("volume_A3_atom")] for row in rows]
    assert volumes[0] == pytest.approx(atoms.get_volume() / len(atoms))
    assert volumes[-1] != volumes[0]
    assert all(math.isfinite(row[tool.COLUMNS.index("pressure_GPa")]) for row in rows)
    assert "D3" in json.dumps(simulation.metadata(), default=str)


def drifting_runs(rng, slopes, n=2000):
    """Runs whose V grows linearly (slope per ps per engine, seeds scattered by 10 %) plus AR(1) noise."""
    runs = []
    for engine, slope, seeds in (("ase", slopes[0], (101, 102, 103)), ("fastmace", slopes[1], (1, 2, 3))):
        for seed in seeds:
            time_ps = np.arange(1, n + 1) * 0.01
            rate = slope * (1 + 0.1 * rng.normal())
            data = dict(time_ps=time_ps, temperature_K=np.full(n, 600.0),
                        volume_A3_atom=32.0 + rate * time_ps + ar1(n, 0.9, 0.5, 0.0, rng),
                        pressure_GPa=ar1(n, 0.5, 0.3, 0.0, rng), potential_eV_atom=ar1(n, 0.7, 0.002, -4.4, rng))
            runs.append(({"engine": engine, "seed": seed, "temperature_K": 600.0, "equilibration_ps": 0.0}, data))
    return runs


def test_a_drifting_volume_is_judged_on_its_drift():
    rng = np.random.default_rng(6)
    entry = tool.analyse(drifting_runs(rng, (0.8, 0.8)))["600K"]
    assert not entry["stationary"] and entry["judged_on"] == "V drift and windowed V means"
    assert entry["passed"], entry["drift_comparison"]
    assert entry["engines"]["ase"]["volume_drift"]["mean"] == pytest.approx(0.8, rel=0.2)
    assert len(entry["engines"]["ase"]["volume_windows"]) == tool.WINDOWS
    faster = tool.analyse(drifting_runs(rng, (0.8, 1.6)))["600K"]
    assert not faster["passed"] and not faster["drift_comparison"]["volume_drift"]["passed"]


def test_stationary_runs_are_judged_on_their_means():
    entry = tool.analyse(synthetic_runs(np.random.default_rng(7)))["300K"]
    assert entry["stationary"] and entry["judged_on"] == "stationary means"


def test_kinetic_extras_split_off_the_centre_of_mass():
    masses = np.array([1.0, 3.0])
    momenta = np.array([[2.0, 0.0, 0.0], [6.0, 0.0, 0.0]])  # both move at v = 2: all kinetic energy is COM
    step, kinetic, com = tool.kinetic_extras(5, masses, momenta)
    assert step == 5 and kinetic == pytest.approx(8.0) and com == pytest.approx(8.0)
    _, kinetic, com = tool.kinetic_extras(0, masses, np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]))
    assert kinetic == pytest.approx(4.5 + 1.5) and com == pytest.approx(0.0)


def test_both_sides_report_extras_equal_to_their_kinetic_energy(tmp_path, monkeypatch):
    """The estimator check: on both sides the sampled kinetic energy is the momenta's (same point, same sum)."""
    pytest.importorskip("cuequivariance_torch")
    pytest.importorskip("torch_dftd")
    import torch
    from ase.calculators.mixing import SumCalculator

    from _support.mc_fixtures import cpu_planner, tiny_loaded
    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.reference.d3 import torch_dftd3_calculator

    cpu_planner(monkeypatch)
    path = tmp_path / "tiny-mh1-11-17.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    settings = {**tool.SETTINGS, "head": None}
    for side in ("ase", "fastmace"):
        rows, extras = [], []
        atoms = nacl(1, rattle=0.05, seed=3)
        if side == "ase":
            loaded = tiny_loaded(tmp_path, (11, 17))
            calculator = SumCalculator([loaded.calculator, torch_dftd3_calculator(tool.d3_record())])
            tool.run_ase(atoms, calculator, 300.0, 5, 10, 5, rows.append, settings=settings, extra=extras.append)
        else:
            tool.fastmace_side(path, atoms, 300.0, 5, 10, 5, rows.append, settings=settings, extra=extras.append,
                               precision="float64")
        assert [row[0] for row in extras] == [row[0] for row in rows] == [0, 5, 10]
        for row, extra in zip(rows, extras, strict=True):
            kinetic = row[tool.COLUMNS.index("kinetic_eV_atom")] * len(atoms)
            assert extra[1] == pytest.approx(kinetic, rel=1e-9), side
