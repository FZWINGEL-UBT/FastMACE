"""4-T09: protocol relaxations and quenches on the GPU optimisers, ``fastmace relax`` and ``fastmace quench``.

The optimizer names (conductor, 23 October 2026): bare ``"LBFGS"`` and ``"BFGS"`` keep meaning ASE's, so
existing protocol files are unchanged; ``"ase-LBFGS"`` and ``"ase-BFGS"`` name them explicitly, and
``"gpu-LBFGS"`` and ``"gpu-FIRE"`` are ``fm.optimize``'s on the graph engine. The records hold the resolved
name. The default stays ``"ase-LBFGS"`` until 4-T08 passes (decision 47).

One GPU relaxer relaxes every frame of a quench on one engine (``Simulation.update(cell=, positions=)``),
each frame as a fresh relaxation would. On the CPU the "GPU" optimisers run on the eager CPU engine.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from fastmace import cli
from fastmace.protocols import PROFILES, profiles, quench, runner

NA_CL = (11, 17)


# --------------------------------------------------------------------------- names (no model)


def test_names_resolve_and_bare_names_stay_ase():
    assert profiles.DEFAULT_OPTIMIZER == "ase-LBFGS"
    assert profiles.resolve_optimizer("LBFGS") == "ase-LBFGS"
    assert profiles.resolve_optimizer("BFGS") == "ase-BFGS"
    for name in ("ase-LBFGS", "ase-BFGS", "gpu-LBFGS", "gpu-FIRE"):
        assert profiles.resolve_optimizer(name) == name
    assert set(profiles.GPU_OPTIMIZERS) == {"gpu-LBFGS", "gpu-FIRE"}
    for name in ("FIRE", "gpu-BFGS", "lbfgs", None, 1):
        with pytest.raises(ValueError, match="optimizer must be one of"):
            profiles.resolve_optimizer(name)


def test_profiles_and_stages_record_the_resolved_name():
    anneal = PROFILES["anneal"]
    assert anneal.relax_optimizer == "ase-LBFGS" and anneal.quench["optimizer"] == "ase-LBFGS"
    stages, _ = runner.split_stages({"relax": {"fmax": 0.05}})
    assert stages["relax"]["optimizer"] == profiles.DEFAULT_OPTIMIZER
    stages, _ = runner.split_stages({"relax": {"fmax": 0.05, "optimizer": "gpu-FIRE"},
                                     "quench": {**dict(anneal.quench), "optimizer": "BFGS"}})
    assert stages["relax"]["optimizer"] == "gpu-FIRE" and stages["quench"]["optimizer"] == "ase-BFGS"
    with pytest.raises(ValueError, match="optimizer must be one of"):
        runner.split_stages({"relax": {"fmax": 0.05, "optimizer": "FIRE"}})


def test_a_gpu_quench_needs_a_relaxer_not_a_calculator(tmp_path):
    from fastmace.protocols import ramp

    resolved = ramp(0.0, 800.0, ps=0.008).resolve(1.0)
    _fake_run(tmp_path, resolved, 4)
    settings = {"min_K": 400.0, "bin_K": 200.0, "per_bin": 1, "fmax": 0.05, "optimizer": "gpu-LBFGS",
                "max_steps": 5}
    with pytest.raises(ValueError, match="gpu-LBFGS"):
        quench.quench_run(tmp_path, _lj(), resolved, 4, 1.0, settings, log=lambda text: None)


# --------------------------------------------------------------------------- quench directories (Lennard-Jones)


def _lj():
    from ase.calculators.lj import LennardJones

    return LennardJones(sigma=2.5, epsilon=0.05, rc=6.0, smooth=True)


def _fake_run(directory, resolved, frame_every):
    """A finished run's chunked output, as ``test_swap_anneal_protocol._fake_run``."""
    from ase.io.trajectory import Trajectory

    from _support.structures import nacl
    from fastmace.io.output import chunk_paths

    directory.mkdir(parents=True, exist_ok=True)
    steps = list(range(0, resolved.total_steps + 1, frame_every))
    thermo, trajectory = chunk_paths(directory, 0)
    with thermo.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time_ps", "phase", "target_K", "temperature_K", "potential_eV_atom"])
        for step in steps:
            writer.writerow([step, 0.0, resolved.phase(step), resolved.target(step), resolved.target(step), -1.0])
    with Trajectory(str(trajectory), "w") as handle:
        for step in steps:
            handle.write(nacl(repeat=1, rattle=0.05, seed=step))


def test_a_quench_continues_only_with_its_own_settings(tmp_path, monkeypatch):
    from fastmace.protocols import ramp

    resolved = (ramp(0.0, 800.0, ps=0.04) + ramp(800.0, 400.0, ps=0.04)).resolve(1.0)
    _fake_run(tmp_path, resolved, 4)
    settings = {"min_K": 400.0, "bin_K": 200.0, "per_bin": 1, "fmax": 0.05, "optimizer": "LBFGS", "max_steps": 200}
    real, calls = quench.relax, []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(quench, "relax", failing)
    with pytest.raises(KeyboardInterrupt):
        quench.quench_run(tmp_path, _lj(), resolved, 4, 1.0, settings, log=lambda text: None)
    monkeypatch.setattr(quench, "relax", real)
    recorded = json.loads((tmp_path / "quench" / quench.SETTINGS).read_text(encoding="utf-8"))
    assert recorded["optimizer"] == "ase-LBFGS"  # the resolved name
    with pytest.raises(ValueError, match="ase-BFGS"):
        quench.quench_run(tmp_path, _lj(), resolved, 4, 1.0, {**settings, "optimizer": "BFGS"})
    # The same optimizer under its explicit name is the same quench.
    result = quench.quench_run(tmp_path, _lj(), resolved, 4, 1.0, {**settings, "optimizer": "ase-LBFGS"},
                               log=lambda text: None)
    assert result["frames"] == 5 and result["settings"]["optimizer"] == "ase-LBFGS"


def test_a_second_quench_goes_into_its_own_directory(tmp_path):
    from fastmace.protocols import ramp

    resolved = ramp(0.0, 800.0, ps=0.016).resolve(1.0)
    _fake_run(tmp_path, resolved, 4)
    settings = {"min_K": 400.0, "bin_K": 200.0, "per_bin": 1, "fmax": 0.05, "optimizer": "LBFGS", "max_steps": 200}
    first = quench.quench_run(tmp_path, _lj(), resolved, 4, 1.0, settings, log=lambda text: None)
    second = quench.quench_run(tmp_path, _lj(), resolved, 4, 1.0, {**settings, "optimizer": "BFGS"},
                               name="quench-bfgs", log=lambda text: None)
    assert (tmp_path / "quench-bfgs" / quench.SUMMARY).is_file()
    assert first["settings"]["optimizer"] == "ase-LBFGS" and second["settings"]["optimizer"] == "ase-BFGS"
    assert quench.read_rows(tmp_path / "quench" / quench.ROWS)[0]["frame"] == \
        quench.read_rows(tmp_path / "quench-bfgs" / quench.ROWS)[0]["frame"]
    for bad in ("", "../x", "a b", "quench/x"):
        with pytest.raises(ValueError, match="name"):
            quench.quench_directory(bad)
    assert quench.quench_directory(None) == "quench" and quench.quench_directory("gpu") == "quench-gpu"


# --------------------------------------------------------------------------- the GPU relaxer (tiny model, CPU)


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


def _model(model_path):
    from fastmace.model.loading import load_pristine

    return load_pristine(model_path, trusted=True)


def _frame(seed, scale=1.0):
    from _support.structures import nacl

    atoms = nacl(repeat=1, rattle=0.08, seed=seed)
    atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
    return atoms


@pytest.mark.parametrize("name", ["gpu-LBFGS", "gpu-FIRE"])
def test_one_gpu_relaxer_relaxes_each_frame_as_a_fresh_one(model_path, cpu_planner, tmp_path, name):
    model = _model(model_path)
    frames = [_frame(3), _frame(5, 1.01), _frame(7, 0.99)]
    shared = quench.relaxer(name, model, precision="float64")
    try:
        results = [shared(frame, 0.5, 50, logfile=tmp_path / f"{index}.log") for index, frame in enumerate(frames)]
    finally:
        shared.close()
    assert shared.builds == 1
    for frame, (relaxed, summary) in zip(frames, results, strict=True):
        alone = quench.relaxer(name, model, precision="float64")
        try:
            fresh, expected = alone(frame, 0.5, 50)
        finally:
            alone.close()
        assert summary["optimizer"] == name and summary["fixed_cell"] is True
        assert summary["steps"] == expected["steps"] and summary["converged"] == expected["converged"]
        np.testing.assert_array_equal(relaxed.positions, fresh.positions)
        np.testing.assert_array_equal(relaxed.cell.array, frame.cell.array)
        assert "energies_eV" not in summary and "fmax_history_eV_A" not in summary  # they go to the log
    log = (tmp_path / "0.log").read_text(encoding="utf-8").splitlines()
    assert log[0].split() == ["Step", "Energy", "fmax"] and len(log) == results[0][1]["steps"] + 2


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_one_gpu_relaxer_with_d3_takes_each_frame_cell(model_path, cpu_planner):
    """With D3 the new cell reaches the D3 list too (the 4-T09 Modal smoke: ``D3Term`` had no ``reset_cell``)."""
    from fastmace.longrange.d3 import D3

    model = _model(model_path)
    frames = [_frame(3), _frame(5, 1.03)]
    shared = quench.relaxer("gpu-LBFGS", model, precision="float64", dispersion=D3(functional="r2scan"))
    try:
        results = [shared(frame, 0.5, 30) for frame in frames]
    finally:
        shared.close()
    assert shared.builds == 1
    alone = quench.relaxer("gpu-LBFGS", model, precision="float64", dispersion=D3(functional="r2scan"))
    try:
        fresh, expected = alone(frames[1], 0.5, 30)
    finally:
        alone.close()
    relaxed, summary = results[1]
    assert summary["steps"] == expected["steps"]
    assert summary["initial_eV_atom"] == pytest.approx(expected["initial_eV_atom"], abs=1e-12)
    np.testing.assert_allclose(relaxed.positions, fresh.positions, rtol=0, atol=1e-12)


def test_the_gpu_relaxer_reaches_the_ase_minimum(model_path, cpu_planner):
    from fastmace.ase.calculator import FastMACECalculator

    model = _model(model_path)
    atoms = _frame(3)
    gpu = quench.relaxer("gpu-LBFGS", model, precision="float64")
    try:
        relaxed, summary = gpu(atoms, 0.05, 2000)
    finally:
        gpu.close()
    reference, expected = quench.relax(atoms, FastMACECalculator(model, precision="float64"), 0.05, 2000,
                                       optimizer="ase-LBFGS")
    assert summary["converged"] and expected["converged"]
    assert summary["energy_eV_atom"] == pytest.approx(expected["energy_eV_atom"], abs=1e-4)


def _tiny_protocol(tmp_path, model_path, optimizer, npt=False):
    from _support.structures import nacl

    structure = tmp_path / "nacl.extxyz"
    nacl(repeat=1, rattle=0.05, seed=3).write(structure)
    config = tmp_path / "tiny.toml"
    ensemble = 'ensemble = "NPT"\npressure_GPa = 0.0\n' if npt else ""
    config.write_text(
        f'structure = "{structure.as_posix()}"\nmodel = "{model_path.replace(chr(92), "/")}"\ntrusted = true\n'
        f'timestep_fs = 1.0\nseed = 7\n{ensemble}'
        '[[schedule]]\nkind = "ramp"\nstart_K = 0.0\nend_K = 600.0\nps = 0.012\n'
        '[[schedule]]\nkind = "ramp"\nstart_K = 600.0\nend_K = 300.0\nps = 0.012\n'
        '[output]\ntrajectory_every = 2\nthermo_every = 2\nrestart_every = 100\n'
        f'[relax]\nfmax = 0.5\noptimizer = "{optimizer}"\nmax_steps = 10\n'
        f'[quench]\nmin_K = 300.0\nbin_K = 150.0\nper_bin = 1\nfmax = 0.5\noptimizer = "{optimizer}"\n'
        'max_steps = 5\n',
        encoding="utf-8")
    return config


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_tiny_npt_protocol_relaxes_and_quenches_on_one_gpu_engine(model_path, cpu_planner, tmp_path, capsys):
    from ase.io.trajectory import Trajectory

    from fastmace.io.output import read_frames

    config = _tiny_protocol(tmp_path, model_path, "gpu-LBFGS", npt=True)
    out = tmp_path / "run"
    assert cli.main(["run", str(config), "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "relaxation: gpu-LBFGS" in printed
    record = json.loads((out / runner.PROTOCOL_RECORD).read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["relax"]["optimizer"] == record["quench"]["optimizer"] == "gpu-LBFGS"
    assert record["relaxation"]["optimizer"] == "gpu-LBFGS"
    assert (out / runner.RELAXATION_LOG).read_text(encoding="utf-8").split()[:3] == ["Step", "Energy", "fmax"]
    result = record["quench_result"]
    assert result["settings"]["optimizer"] == "gpu-LBFGS" and result["engines"] == 1
    # Each frame at its own NPT cell, on the one engine.
    frames = list(read_frames(out))
    rows = quench.read_rows(out / "quench" / quench.ROWS)
    with Trajectory(str(out / "quench" / quench.QUENCHED)) as quenched:
        for row, after in zip(rows, quenched, strict=True):
            np.testing.assert_array_equal(after.cell[:], frames[row["frame"]].cell[:])
    assert len({row["volume_A3_atom"] for row in rows}) > 1

    # fastmace quench: the same frames again with ASE's LBFGS, into quench-ase/; the protocol's stays.
    assert cli.main(["quench", str(out), "--into", "ase", "--optimizer", "ase-LBFGS"]) == 0
    capsys.readouterr()
    other = json.loads((out / "quench-ase" / quench.SUMMARY).read_text(encoding="utf-8"))
    assert other["settings"]["optimizer"] == "ase-LBFGS" and other["settings"]["min_K"] == 300.0
    assert [r["frame"] for r in quench.read_rows(out / "quench-ase" / quench.ROWS)] == [r["frame"] for r in rows]
    # A complete quench prints its summary and relaxes nothing.
    assert cli.main(["quench", str(out)]) == 0
    assert "complete" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fastmace_quench_on_a_plain_run(model_path, cpu_planner, tmp_path, capsys):
    from _support.structures import nacl
    from fastmace import quick
    from fastmace.io.output import Output
    from fastmace.protocols import ramp

    out = tmp_path / "plain"
    schedule = ramp(0.0, 600.0, ps=0.008) + ramp(600.0, 300.0, ps=0.008)
    run = quick.md(nacl(repeat=1, rattle=0.05, seed=3), model=model_path, trusted=True, schedule=schedule,
                   timestep_fs=1.0, seed=1, output=Output(str(out), trajectory_every=2, thermo_every=2))
    run.simulation.save_restart(out)
    run.simulation.close()
    args = ["quench", str(out), "--min-K", "300", "--bin-K", "150", "--per-bin", "1", "--fmax", "0.5",
            "--max-steps", "5", "--optimizer", "gpu-FIRE"]
    assert cli.main(args) == 0
    capsys.readouterr()
    summary = json.loads((out / "quench" / quench.SUMMARY).read_text(encoding="utf-8"))
    assert summary["settings"] == {**summary["settings"], "optimizer": "gpu-FIRE", "per_bin": 1, "max_steps": 5}
    assert summary["engines"] == 1 and summary["frames"] == len(quench.read_rows(out / "quench" / quench.ROWS))


def test_fastmace_quench_usage(tmp_path, capsys):
    assert cli.main(["quench", str(tmp_path)]) == 2
    assert "restart" in capsys.readouterr().err
    assert cli.main(["quench", str(tmp_path), "--into", "../x"]) == 2
    assert "name" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["quench", str(tmp_path), "--optimizer", "FIRE"])
    capsys.readouterr()


# --------------------------------------------------------------------------- fastmace relax


def test_relax_is_a_command():
    assert cli.COMMANDS["relax"] == "fastmace.cli.relax" and cli.COMMANDS["quench"] == "fastmace.cli.quench"


def test_fastmace_relax_usage(tmp_path, capsys):
    structure = tmp_path / "a.extxyz"
    assert cli.main(["relax", str(structure), "--model", "x", "--pressure", "1"]) == 2
    assert "--cell-filter" in capsys.readouterr().err
    assert cli.main(["relax", str(structure), "--model", "x", "--fmax", "0"]) == 2
    assert "--fmax" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["relax", str(structure), "--model", "x", "--optimizer", "FIRE"])
    capsys.readouterr()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("optimizer, cell", [("ase-LBFGS", False), ("gpu-LBFGS", False), ("ase-LBFGS", True),
                                             ("gpu-FIRE", True)])
def test_fastmace_relax_writes_the_structure_and_its_summary(model_path, cpu_planner, tmp_path, capsys,
                                                            optimizer, cell):
    from ase.io import read

    structure = tmp_path / "nacl.extxyz"
    _frame(3, 1.02).write(structure)
    output = tmp_path / "relaxed.extxyz"
    # At 5 GPa the cell rows start at 0.70 eV/A; the random tiny model's landscape is slow for ASE's LBFGS on
    # the filter below 0.5, so the cell cases stop at 0.6 (after some steps that moved the cell).
    fmax = 0.6 if cell else 0.3
    args = ["relax", str(structure), "--model", model_path, "--trusted", "--fmax", str(fmax), "--steps", "300",
            "--optimizer", optimizer, "--output", str(output)]
    if cell:
        args += ["--cell-filter", "frechet", "--pressure", "5"]
    assert cli.main(args) == 0, capsys.readouterr().err
    printed = capsys.readouterr().out
    assert optimizer in printed and str(output) in printed
    relaxed = read(output)
    summary = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["optimizer"] == optimizer and summary["converged"] and summary["fmax_eV_A"] <= fmax
    assert summary["fixed_cell"] is (not cell)
    assert output.with_suffix(".log").is_file()
    start = read(structure)
    if cell:
        assert summary["scalar_pressure_GPa"] == 5.0 and summary["steps"] > 0
        assert not np.allclose(relaxed.cell.array, start.cell.array)
    else:
        np.testing.assert_array_equal(relaxed.cell.array, start.cell.array)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fastmace_relax_not_converged_is_an_error(model_path, cpu_planner, tmp_path, capsys):
    structure = tmp_path / "nacl.extxyz"
    _frame(3).write(structure)
    output = tmp_path / "relaxed.extxyz"
    assert cli.main(["relax", str(structure), "--model", model_path, "--trusted", "--fmax", "1e-6", "--steps", "2",
                     "--optimizer", "gpu-LBFGS", "--output", str(output)]) == 1
    assert "did not reach fmax" in capsys.readouterr().err
    assert json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["converged"] is False
