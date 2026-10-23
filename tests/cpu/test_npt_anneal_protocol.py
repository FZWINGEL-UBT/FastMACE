"""3b-T07: the NPT melt-quench protocol as data (owner decision D1 = SPEC §18 decision 40).

NbCl5 3x3x3 on MH-1 matpes_r2scan with D3-BJ, hydrostatic NPT at 0 GPa along the anneal profile at
1 fs, with the library's barostat defaults, a fixed-cell relaxation first and the anneal quench.
"""

from __future__ import annotations

import pytest

from fastmace.cli import run as run_command
from fastmace.protocols import PROFILES, ScheduleState, runner

FULL, SHORT = "npt_anneal_NbCl5", "npt_anneal_NbCl5_short"


def _data(name):
    path, shipped = runner.protocol_file(name)
    assert shipped and path.parent == runner.DATA
    return run_command.parse_toml(path.read_text(encoding="utf-8"))


def test_the_shipped_protocol_is_decision_40():
    from fastmace.config import EngineConfig
    from fastmace.io.output import Output

    data = _data(FULL)
    stages, rest = runner.split_stages(data)
    # Frames every 100 fs (not main.py's 10 fs, decision 36) keep the 1944-atom trajectory to a few
    # hundred MB; thermo, with the volume, stays at 10 fs.
    assert rest["output"] == {"trajectory_every": 100, "thermo_every": 10, "restart_every": 10_000}
    rest["output"] = run_command._with_output(rest, None)
    structure, options = run_command.md_arguments(rest)
    assert (runner.DATA / structure).is_file()
    assert options["model"] == "mh1" and options["head"] == "matpes_r2scan"
    assert options["dispersion"] == "d3bj"
    assert options["repeat"] == (3, 3, 3) and options["timestep_fs"] == 1.0
    assert options["schedule"] == PROFILES["anneal"].schedule
    # Hydrostatic NPT at 0 GPa with the library's barostat defaults: no integrator table.
    assert (options["ensemble"], options["pressure_GPa"], options["cell_mode"]) == ("NPT", 0.0, "hydrostatic")
    assert "integrator" not in options and "mc" not in options
    assert options["engine"] == EngineConfig(backend="hybrid")
    assert options["output"] == Output("NbCl5_R2SCAN.fastmace", trajectory_every=100, thermo_every=10,
                                       restart_every=10_000)
    anneal = PROFILES["anneal"]
    assert stages["relax"] == {"fmax": anneal.relax_fmax, "optimizer": anneal.relax_optimizer, "max_steps": 5000}
    assert stages["quench"] == dict(anneal.quench)


def test_the_short_protocol_differs_only_in_its_schedule():
    full, short = _data(FULL), _data(SHORT)
    assert {key: value for key, value in short.items() if key != "schedule"} == \
        {key: value for key, value in full.items() if key != "schedule"}
    position = ScheduleState(run_command._schedule(short["schedule"]), 1.0)
    assert position.max_target() == 1000.0 and position.resolved.target(position.total_steps) == 250.0
    assert position.total_steps == 27_000  # 2 ps up, 25 ps down: a tenth of the full 270,000 steps


def test_the_quench_has_frames_in_every_bin_at_100_fs():
    from fastmace.protocols import quench

    resolved = ScheduleState(PROFILES["anneal"].schedule, 1.0).resolved
    picks = quench.picks(resolved, 100, PROFILES["anneal"].quench)
    assert len(picks) == 21  # 2 per 100 K bin from 500 K on each ramp, and the final frame


def test_the_shipped_cif_is_the_r2scan_benchmark():
    from ase.io import read

    atoms = read(runner.DATA / "NbCl5_R2SCAN.cif")
    assert atoms.get_chemical_formula() == "Cl60Nb12" and len(atoms) == 72
    assert len(atoms.repeat((3, 3, 3))) == 1944
    a, b, c, *_ = atoms.cell.cellpar()
    assert (a, b, c) == pytest.approx((18.9641, 18.1777, 6.0072), abs=1e-4)
    assert "G:/Margraf/ResearchModule/Structures/Benchmark/R2SCAN/NbCl5.cif" in \
        (runner.DATA / "SOURCES.md").read_text(encoding="utf-8")


def test_protocol_names():
    assert {FULL, SHORT} <= set(runner.protocol_names())


# --------------------------------------------------------------------------- a tiny NPT + D3 protocol on the CPU

NA_CL = (11, 17)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    import torch

    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return str(path)


@pytest.fixture
def cpu_planner(monkeypatch):
    from _support.mc_fixtures import cpu_planner

    return cpu_planner(monkeypatch)


def _tiny_npt_protocol(tmp_path, model_path):
    from _support.structures import nacl

    structure = tmp_path / "nacl.extxyz"
    nacl(repeat=1, rattle=0.05, seed=3).write(structure)
    config = tmp_path / "tiny_npt.toml"
    # 1 fs steps: 0 -> 600 K in 12 steps, then 600 -> 300 K in 12; NPT at 0 GPa with D3 (r2scan's BJ set).
    config.write_text(
        f'structure = "{structure.as_posix()}"\nmodel = "{model_path.replace(chr(92), "/")}"\ntrusted = true\n'
        'timestep_fs = 1.0\nseed = 7\nensemble = "NPT"\npressure_GPa = 0.0\n'
        '[dispersion]\nfunctional = "r2scan"\n'
        '[[schedule]]\nkind = "ramp"\nstart_K = 0.0\nend_K = 600.0\nps = 0.012\n'
        '[[schedule]]\nkind = "ramp"\nstart_K = 600.0\nend_K = 300.0\nps = 0.012\n'
        '[output]\ntrajectory_every = 2\nthermo_every = 2\nrestart_every = 100\n'
        '[relax]\nfmax = 0.5\noptimizer = "LBFGS"\nmax_steps = 10\n'
        '[quench]\nmin_K = 300.0\nbin_K = 150.0\nper_bin = 1\nfmax = 0.5\noptimizer = "LBFGS"\nmax_steps = 5\n',
        encoding="utf-8")
    return config


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_tiny_npt_d3_protocol_records_d3_and_quenches_at_the_frame_cells(model_path, cpu_planner, tmp_path,
                                                                           capsys):
    import json

    import numpy as np
    from ase.io import read
    from ase.io.trajectory import Trajectory

    from fastmace import cli
    from fastmace.ase.calculator import FastMACECalculator
    from fastmace.io.output import read_frames
    from fastmace.longrange.d3 import D3
    from fastmace.protocols import quench

    config = _tiny_npt_protocol(tmp_path, model_path)
    out = tmp_path / "run"
    assert cli.main(["run", str(config), "--output", str(out), "--steps", "10"]) == 0
    assert cli.main(["run", "--resume", str(out)]) == 0
    capsys.readouterr()
    record = json.loads((out / runner.PROTOCOL_RECORD).read_text(encoding="utf-8"))
    assert record["status"] == "complete" and len(record["pieces"]) == 2
    assert "functional=r2scan" in record["dispersion"]

    # D3's counters per piece and over the run (3b-T06's cross-check), beside the MACE list's.
    for piece in record["pieces"]:
        counts = piece["counters"]
        assert set(counts) == set(runner.PIECE_COUNTERS)
        assert counts["d3_builds"] >= 1 and counts["d3_build_s"] > 0 and counts["d3_rebuilds"] >= 0
    total = record["counters"]
    assert total["d3_builds"] == sum(p["counters"]["d3_builds"] for p in record["pieces"])
    assert total["d3_build_s"] == pytest.approx(sum(p["counters"]["d3_build_s"] for p in record["pieces"]))

    # The relaxation ran on MACE plus D3: its energy is the D3 calculator's, not MACE's alone.
    from fastmace.model.loading import load_pristine

    model = load_pristine(model_path, trusted=True)
    relaxed = read(out / runner.RELAXED)
    with_d3 = FastMACECalculator(model, dispersion=D3(functional="r2scan"))
    alone = FastMACECalculator(model)
    e_d3 = with_d3.get_potential_energy(relaxed.copy()) / len(relaxed)
    e_alone = alone.get_potential_energy(relaxed.copy()) / len(relaxed)
    assert record["relaxation"]["energy_eV_atom"] == pytest.approx(e_d3, abs=1e-5)
    assert abs(e_d3 - e_alone) > 1e-3

    # Each quench frame keeps its own NPT cell, and the rows record its volume.
    frames = list(read_frames(out))
    rows = quench.read_rows(out / "quench" / "quench.csv")
    with Trajectory(str(out / "quench" / quench.CAPTURED)) as captured, \
            Trajectory(str(out / "quench" / quench.QUENCHED)) as quenched:
        for row, before, after in zip(rows, captured, quenched, strict=True):
            np.testing.assert_array_equal(before.cell[:], frames[row["frame"]].cell[:])
            np.testing.assert_array_equal(after.cell[:], before.cell[:])
            assert row["volume_A3_atom"] == pytest.approx(before.get_volume() / len(before))
    assert len({row["volume_A3_atom"] for row in rows}) > 1  # the cells differ: NPT moved them
    summary = json.loads((out / "quench" / quench.SUMMARY).read_text(encoding="utf-8"))
    assert summary["settings"]["fixed_cell"] is True and summary["settings"]["cell"] == "frame"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_diverging_npt_protocol_records_the_stopped_piece(model_path, cpu_planner, tmp_path, capsys):
    import json

    from _support.structures import nacl
    from fastmace import cli

    structure = tmp_path / "nacl.extxyz"
    nacl(repeat=2, rattle=0.02, seed=1).write(structure)
    config = tmp_path / "tension.toml"
    # Tension at -2 GPa expands the cell past V/V0 = 1.02 within the 24 steps.
    config.write_text(
        f'structure = "{structure.as_posix()}"\nmodel = "{model_path.replace(chr(92), "/")}"\ntrusted = true\n'
        'seed = 7\n'
        '[integrator]\ntype = "LangevinBAOAB"\ntimestep_fs = 1.0\ntemperature_K = 0.0\npressure_GPa = -2.0\n'
        'barostat_tau_fs = 100.0\nmax_volume_ratio = 1.02\n'
        '[[schedule]]\nkind = "ramp"\nstart_K = 0.0\nend_K = 300.0\nps = 0.024\n'
        '[output]\ntrajectory_every = 2\nthermo_every = 2\nrestart_every = 100\n'
        '[quench]\nmin_K = 100.0\nbin_K = 100.0\nper_bin = 1\nfmax = 0.5\noptimizer = "LBFGS"\nmax_steps = 5\n',
        encoding="utf-8")
    out = tmp_path / "run"
    assert cli.main(["run", str(config), "--output", str(out)]) == 1
    assert "V/V0" in capsys.readouterr().err
    record = json.loads((out / runner.PROTOCOL_RECORD).read_text(encoding="utf-8"))
    assert record["status"] == "cell diverged" and "V/V0" in record["stop"]
    assert record["timestep_fs"] == 1.0  # the integrator record's, not fm.md's default
    (piece,) = record["pieces"]
    assert piece["first_step"] == 0 and 0 < piece["last_step"] < 24
    assert piece["counters"]["neighbour_rebuilds"] is not None and record["counters"] == piece["counters"]
    assert record["quench_result"] is None and (out / "restart.npz").is_file()
    assert cli.main(["run", "--resume", str(out)]) == 2  # stopped: not resumable
    assert "cell diverged" in capsys.readouterr().err
