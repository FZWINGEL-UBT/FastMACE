"""3-T14: the swap-MC annealing protocol (SPEC §10.5; owner decision D3 = SPEC §18 decision 33).

The shipped protocol as data, the ported relaxation, frame selection and quench
(``fastmace.protocols.quench``), ``fastmace run`` with ``[relax]``/``[quench]`` tables, ``--steps`` and
``--resume``, and a tiny CPU protocol run in pieces against the same run in one go.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from fastmace import cli
from fastmace.cli import run as run_command
from fastmace.protocols import PROFILES, ScheduleState, quench, runner

NA_CL = (11, 17)


# --------------------------------------------------------------------------- the shipped protocol


def _shipped():
    path, shipped = runner.protocol_file("swap_anneal_NaNbOCl4")
    return path, shipped, run_command.parse_toml(path.read_text(encoding="utf-8"))


def test_the_shipped_protocol_is_decision_33():
    from fastmace import mc
    from fastmace.config import EngineConfig
    from fastmace.io.output import Output

    path, shipped, data = _shipped()
    assert shipped and path.parent == runner.DATA
    stages, rest = runner.split_stages(data)
    assert rest["output"] == {"trajectory_every": 10, "thermo_every": 10, "restart_every": 10_000}
    rest["output"] = run_command._with_output(rest, None)  # fm.md's default directory, as fastmace run fills it
    structure, options = run_command.md_arguments(rest)
    assert (runner.DATA / structure).is_file()
    assert options["model"] == "mh1" and options["head"] == "matpes_r2scan"
    assert options["repeat"] == (3, 3, 3) and options["timestep_fs"] == 1.0
    assert options["schedule"] == PROFILES["anneal"].schedule  # 0 -> 1000 K in 20 ps, then 3 K/ps to 250 K
    assert options["mc"] == (mc.Swap(pairs=[("O", "Cl")], every=100, attempts=50, velocities="exchange"),)
    assert options["min_mc_temperature_K"] == 250.0
    assert options["engine"] == EngineConfig(backend="hybrid")
    # Physical intervals at 1 fs (D6): frames and thermo every 10 fs, a restart every 10 ps.
    assert options["output"] == Output("NaNbOCl4_I4m_Ma2026.fastmace", trajectory_every=10, thermo_every=10,
                                       restart_every=10_000)
    # The anneal profile's relaxation and quench.
    anneal = PROFILES["anneal"]
    assert stages["relax"] == {"fmax": anneal.relax_fmax, "optimizer": anneal.relax_optimizer, "max_steps": 5000}
    assert stages["quench"] == dict(anneal.quench)


def test_the_short_protocol_differs_only_in_its_schedule():
    _, _, full = _shipped()
    path, shipped = runner.protocol_file("swap_anneal_NaNbOCl4_short")
    short = run_command.parse_toml(path.read_text(encoding="utf-8"))
    assert shipped
    assert {key: value for key, value in short.items() if key != "schedule"} == \
        {key: value for key, value in full.items() if key != "schedule"}
    schedule = run_command._schedule(short["schedule"])
    position = ScheduleState(schedule, 1.0)
    assert position.max_target() == 1000.0 and position.resolved.target(position.total_steps) == 250.0
    assert position.total_steps < 30_000  # every stage, about a tenth of the full 270,000 steps


def test_the_shipped_cif_is_ma_2026_as_given():
    from ase.io import read

    atoms = read(runner.DATA / "NaNbOCl4_I4m_Ma2026.cif")
    assert atoms.get_chemical_formula() == "Cl16Na4Nb4O4" and len(atoms) == 28
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    # Written as P1 with a != b and gamma = 89.52 degrees, not a tetragonal I4/m metric: used as given.
    assert (a, b, c) == pytest.approx((8.304, 9.875, 8.02)) and gamma == pytest.approx(89.52)
    assert len(atoms.repeat((3, 3, 3))) == 756
    assert "G:/Margraf/ResearchModule/Structures/NaMOCl4/NaNbOCl4_I4m_Ma2026.cif" in \
        (runner.DATA / "SOURCES.md").read_text(encoding="utf-8")


def test_protocol_names_and_paths(tmp_path):
    assert {"swap_anneal_NaNbOCl4", "swap_anneal_NaNbOCl4_short"} <= set(runner.protocol_names())
    own = tmp_path / "mine.toml"
    own.write_text("x = 1\n", encoding="utf-8")
    assert runner.protocol_file(own) == (own, False)
    with pytest.raises(ValueError, match="swap_anneal_NaNbOCl4"):
        runner.protocol_file("no_such_protocol")


def test_stage_tables_are_checked():
    assert runner.split_stages({"a": 1}) == ({"relax": None, "quench": None}, {"a": 1})
    with pytest.raises(ValueError, match="relax"):
        runner.split_stages({"relax": {"fmax": 0.01, "optimiser": "LBFGS"}})
    with pytest.raises(ValueError, match="quench"):
        runner.split_stages({"quench": {"min_K": 500.0}})
    stages, _ = runner.split_stages({"relax": {"fmax": 0.05}})
    assert stages["relax"] == {"fmax": 0.05, "optimizer": "gpu-LBFGS", "max_steps": quench.QUENCH_MAX_STEPS}


# --------------------------------------------------------------------------- frame selection


def test_the_anneal_picks_at_1_fs():
    resolved = PROFILES["anneal"].schedule.resolve(1.0)
    chosen = quench.picks(resolved, 10, PROFILES["anneal"].quench)
    # 2 frames in each of the five 100 K bins from 500 K, on each ramp, then the final frame.
    assert [pick["phase"] for pick in chosen] == ["heating"] * 10 + ["cooling"] * 10 + ["final"]
    assert chosen[-1]["step"] == resolved.total_steps == 270_000 and chosen[-1]["target_K"] == 250.0
    assert all(pick["step"] % 10 == 0 and pick["index"] == pick["step"] // 10 for pick in chosen)
    for phase in ("heating", "cooling"):
        bins = [pick["bin"] for pick in chosen if pick["phase"] == phase]
        assert sorted(set(bins)) == ["500-600 K", "600-700 K", "700-800 K", "800-900 K", "900-1000 K"]
        assert all(bins.count(name) == 2 for name in set(bins))
    for pick in chosen[:-1]:
        low, high = (float(t) for t in pick["bin"][:-2].split("-"))
        assert low <= pick["target_K"] <= high


def test_select_frames_is_the_research_function():
    # The picks of langevin/quench.py:select_frames on this table: evenly spaced in each bin, the last bin closed.
    frames = [{"step": s, "target_K": float(s), "phase": "heating"} for s in range(0, 1001, 50)]
    frames += [{"step": 1000 + s, "target_K": 1000.0 - s, "phase": "cooling"} for s in range(50, 1001, 50)]
    chosen = quench.select_frames(frames, 500.0, 250.0, 2, 1000.0)
    assert [(c["step"], c["bin"]) for c in chosen] == [
        (550, "500-750 K"), (650, "500-750 K"), (800, "750-1000 K"), (950, "750-1000 K"),
        (1100, "750-1000 K"), (1200, "750-1000 K"), (1350, "500-750 K"), (1450, "500-750 K")]


# --------------------------------------------------------------------------- relaxation and quench (no model)


def _lj(atoms):
    from ase.calculators.lj import LennardJones

    return LennardJones(sigma=2.5, epsilon=0.05, rc=6.0, smooth=True)


def test_relax_lowers_the_energy_at_fixed_cell():
    from _support.structures import nacl

    atoms = nacl(repeat=1, rattle=0.1, seed=4)
    relaxed, summary = quench.relax(atoms, _lj(atoms), fmax=0.01, max_steps=500)
    assert summary["converged"] and summary["fmax_eV_A"] <= 0.01
    assert summary["energy_eV_atom"] < summary["initial_eV_atom"]
    assert summary["optimizer"] == "ase-LBFGS" and summary["fixed_cell"]
    assert np.array_equal(relaxed.cell, atoms.cell) and relaxed.calc is None
    assert not np.any(relaxed.get_momenta())
    with pytest.raises(ValueError, match="optimizer"):
        quench.relax(atoms, _lj(atoms), fmax=0.01, optimizer="FIRE")


def _fake_run(directory, resolved, frame_every):
    """A finished run's chunked output: frames every ``frame_every`` steps, thermo rows at the same steps."""
    from ase.io.trajectory import Trajectory

    from _support.structures import nacl
    from fastmace.io.output import chunk_paths

    steps = list(range(0, resolved.total_steps + 1, frame_every))
    half = len(steps) // 2
    for chunk, part in enumerate((steps[:half], steps[half:])):
        thermo, trajectory = chunk_paths(directory, chunk)
        with thermo.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "time_ps", "phase", "target_K", "temperature_K", "potential_eV_atom"])
            for step in part:
                writer.writerow([step, 0.0, resolved.phase(step), resolved.target(step), resolved.target(step) + 1,
                                 -1.0 - step * 1e-4])
        with Trajectory(str(trajectory), "w") as handle:
            for step in part:
                handle.write(nacl(repeat=1, rattle=0.05, seed=step))
    return steps


def test_quench_continues_an_interrupted_quench(tmp_path, monkeypatch):
    from fastmace.protocols import ramp

    resolved = (ramp(0.0, 800.0, ps=0.04) + ramp(800.0, 400.0, ps=0.04)).resolve(1.0)
    settings = {"min_K": 400.0, "bin_K": 200.0, "per_bin": 1, "fmax": 0.05, "optimizer": "LBFGS", "max_steps": 200}
    for name in ("whole", "pieces"):
        (tmp_path / name).mkdir()
        _fake_run(tmp_path / name, resolved, 4)
    calc = _lj(None)
    whole = quench.quench_run(tmp_path / "whole", calc, resolved, 4, 1.0, settings, log=lambda text: None)
    assert whole["frames"] == 5 and whole["heating"]["n"] == 2 and whole["cooling"]["n"] == 2

    real, calls = quench.relax, []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(quench, "relax", failing)
    with pytest.raises(KeyboardInterrupt):
        quench.quench_run(tmp_path / "pieces", calc, resolved, 4, 1.0, settings, log=lambda text: None)
    assert len(quench.read_rows(tmp_path / "pieces" / "quench" / "quench.csv")) == 2
    monkeypatch.setattr(quench, "relax", real)
    lines = []
    pieces = quench.quench_run(tmp_path / "pieces", calc, resolved, 4, 1.0, settings, log=lines.append)
    assert lines[0] == "quench: continuing after 2 of 5 frames"
    first = quench.read_rows(tmp_path / "whole" / "quench" / "quench.csv")
    second = quench.read_rows(tmp_path / "pieces" / "quench" / "quench.csv")
    drop = ("seconds",)
    assert [{k: v for k, v in r.items() if k not in drop} for r in first] == \
        [{k: v for k, v in r.items() if k not in drop} for r in second]
    assert pieces["final_quenched_eV_atom"] == whole["final_quenched_eV_atom"]
    assert first[-1]["phase"] == "final" and first[-1]["step"] == resolved.total_steps
    assert first[0]["md_temperature_K"] == resolved.target(first[0]["step"]) + 1
    # Complete: a further call returns the summary and relaxes nothing.
    monkeypatch.setattr(quench, "relax", None)
    assert quench.quench_run(tmp_path / "pieces", calc, resolved, 4, 1.0, settings)["frames"] == 5


def test_quench_checks_the_frame_count(tmp_path):
    from fastmace.protocols import ramp

    resolved = ramp(0.0, 800.0, ps=0.04).resolve(1.0)
    _fake_run(tmp_path, resolved, 4)
    settings = {"min_K": 400.0, "bin_K": 200.0, "per_bin": 1, "fmax": 0.05, "optimizer": "LBFGS", "max_steps": 5}
    with pytest.raises(ValueError, match="expected 21"):
        quench.quench_run(tmp_path, _lj(None), resolved, 2, 1.0, settings)


# --------------------------------------------------------------------------- fastmace run: usage


def test_run_needs_a_config_or_resume(capsys):
    assert cli.main(["run"]) == 2
    assert "CONFIG.toml or --resume" in capsys.readouterr().err
    assert cli.main(["run", "a.toml", "--resume", "dir"]) == 2
    assert "not both" in capsys.readouterr().err


def test_resume_needs_a_protocol_run(tmp_path, capsys):
    assert cli.main(["run", "--resume", str(tmp_path)]) == 2
    assert "protocol.json" in capsys.readouterr().err


def test_steps_must_be_positive(capsys):
    assert cli.main(["run", "swap_anneal_NaNbOCl4", "--steps", "0"]) == 2
    assert "--steps" in capsys.readouterr().err


def test_a_shipped_protocol_resolves_its_structure_in_the_data_directory(monkeypatch, tmp_path):
    seen = {}

    def start(structure, options, **kwargs):
        seen.update(structure=structure, options=options, **kwargs)
        return {"status": "md"}

    monkeypatch.setattr(runner, "start", start)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", "swap_anneal_NaNbOCl4", "--steps", "100", "--output", "out"]) == 0
    assert seen["structure"] == str(runner.DATA / "NaNbOCl4_I4m_Ma2026.cif")
    assert seen["steps"] == 100 and seen["relax"]["fmax"] == 0.01 and seen["quench"]["per_bin"] == 2
    assert seen["options"]["output"].directory == "out" and seen["options"]["output"].trajectory_every == 10
    assert seen["source"] == "swap_anneal_NaNbOCl4"


def test_a_plain_config_still_goes_to_fm_md(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(run_command, "run_md", lambda structure, options, **kwargs: called.update(s=structure) or 0)
    monkeypatch.setattr(runner, "start", None)
    config = tmp_path / "plain.toml"
    config.write_text('structure = "a.xyz"\nmodel = "mh1"\ntemperature_K = 300.0\nsteps = 10\n', encoding="utf-8")
    assert cli.main(["run", str(config)]) == 0
    assert called["s"] == "a.xyz"


# --------------------------------------------------------------------------- a tiny protocol on the CPU


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
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.planner import minimal

    record = constraints.ProcessRecord()

    def planner(force_field):
        return minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                      process=record)

    monkeypatch.setattr(minimal, "default_planner", planner)


def _tiny_protocol(tmp_path, model_path):
    from _support.structures import nacl

    structure = tmp_path / "nacl.extxyz"
    nacl(repeat=1, rattle=0.05, seed=3).write(structure)
    config = tmp_path / "tiny.toml"
    # 1 fs steps: 0 -> 600 K in 12 steps, then 600 -> 300 K in 12; MC from step 6 (300 K) on.
    config.write_text(
        f'structure = "{structure.as_posix()}"\nmodel = "{model_path.replace(chr(92), "/")}"\ntrusted = true\n'
        'timestep_fs = 1.0\nseed = 7\nmin_mc_temperature_K = 300.0\n'
        '[[schedule]]\nkind = "ramp"\nstart_K = 0.0\nend_K = 600.0\nps = 0.012\n'
        '[[schedule]]\nkind = "ramp"\nstart_K = 600.0\nend_K = 300.0\nps = 0.012\n'
        '[output]\ntrajectory_every = 2\nthermo_every = 2\nrestart_every = 100\n'
        '[[mc]]\nmove = "swap"\npairs = [["Na", "Cl"]]\nevery = 2\nattempts = 2\nvelocities = "exchange"\n'
        '[relax]\nfmax = 0.5\noptimizer = "LBFGS"\nmax_steps = 10\n'
        '[quench]\nmin_K = 300.0\nbin_K = 150.0\nper_bin = 1\nfmax = 0.5\noptimizer = "LBFGS"\nmax_steps = 5\n',
        encoding="utf-8")
    return config


def _frames(directory):
    from fastmace.io.output import read_frames

    return [(frame.numbers.copy(), frame.positions.copy(), frame.get_momenta()) for frame in read_frames(directory)]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_tiny_protocol_in_pieces_equals_one_go(model_path, cpu_planner, tmp_path, capsys):
    config = _tiny_protocol(tmp_path, model_path)
    whole, pieces = tmp_path / "whole", tmp_path / "pieces"
    assert cli.main(["run", str(config), "--output", str(whole)]) == 0
    printed = capsys.readouterr().out
    assert "relaxation:" in printed and "quench:" in printed and "mc swap:" in printed

    assert cli.main(["run", str(config), "--output", str(pieces), "--steps", "5"]) == 0  # before MC attaches
    record = json.loads((pieces / runner.PROTOCOL_RECORD).read_text(encoding="utf-8"))
    assert record["status"] == "md" and record["pieces"][-1]["last_step"] == 5
    assert "fastmace run --resume" in capsys.readouterr().out
    assert cli.main(["run", str(config), "--output", str(pieces)]) == 2  # already a protocol run
    assert "--resume" in capsys.readouterr().err
    assert cli.main(["run", "--resume", str(pieces), "--steps", "9"]) == 0  # attaches MC at step 5
    assert cli.main(["run", "--resume", str(pieces), "--steps", "1000"]) == 0  # the rest (clamped), the quench
    capsys.readouterr()

    first = json.loads((whole / runner.PROTOCOL_RECORD).read_text(encoding="utf-8"))
    second = json.loads((pieces / runner.PROTOCOL_RECORD).read_text(encoding="utf-8"))
    assert first["status"] == second["status"] == "complete"
    assert [(p["first_step"], p["last_step"]) for p in second["pieces"]] == [(0, 5), (5, 14), (14, 24)]
    # Each piece's engine counters (a fresh engine per piece), and their sums over the whole run (3b-T07).
    for record in (first, second):
        rebuilds = [piece["counters"]["neighbour_rebuilds"] for piece in record["pieces"]]
        assert all(isinstance(count, int) and count >= 0 for count in rebuilds), rebuilds
        assert record["counters"]["neighbour_rebuilds"] == sum(rebuilds)
        assert record["counters"]["d3_builds"] is None  # no D3 in this run
    assert first["relaxation"]["energy_eV_atom"] == second["relaxation"]["energy_eV_atom"]
    for (n1, x1, p1), (n2, x2, p2) in zip(_frames(whole), _frames(pieces), strict=True):
        assert np.array_equal(n1, n2) and np.array_equal(x1, x2) and np.array_equal(p1, p2)
    meta1 = json.loads((whole / "metadata.json").read_text(encoding="utf-8"))
    meta2 = json.loads((pieces / "metadata.json").read_text(encoding="utf-8"))
    assert meta1["counters"]["mc"] == meta2["counters"]["mc"]
    counts = meta1["counters"]["mc"]["swap"]
    assert counts["attempted"] > 0 and counts["accepted"] > 0, counts  # the pieces cross accepted swaps
    assert meta1["report"]["mc"]["ranges"] == meta2["report"]["mc"]["ranges"] == [[5, 24]]
    assert second["repeat"] == [1, 1, 1] and second["atoms"] == 8
    rows1 = quench.read_rows(whole / "quench" / "quench.csv")
    rows2 = quench.read_rows(pieces / "quench" / "quench.csv")
    assert [r["quenched_eV_atom"] for r in rows1] == [r["quenched_eV_atom"] for r in rows2]
    assert rows1[-1]["phase"] == "final" and rows1[-1]["step"] == 24
    assert (pieces / "relaxation.log").is_file() and (pieces / "relaxed.extxyz").is_file()
    # A complete protocol run resumes to nothing.
    assert cli.main(["run", "--resume", str(pieces)]) == 0
    assert "complete" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_md_with_steps_plans_for_the_whole_schedule(model_path, cpu_planner, tmp_path):
    # fm.md sizes the workload by the whole schedule's maximum (SPEC §7.7), so the rest of the schedule
    # can run on the same plan later (run(), a resume) instead of being refused above max_temperature_K.
    from _support.structures import nacl
    from fastmace import quick
    from fastmace.protocols import ramp

    schedule = ramp(0.0, 600.0, ps=0.004) + ramp(600.0, 300.0, ps=0.004)
    run = quick.md(nacl(repeat=1, rattle=0.05, seed=3), model=model_path, trusted=True, schedule=schedule,
                   timestep_fs=1.0, steps=2, seed=1, output=str(tmp_path / "run"))
    sim = run.simulation
    assert sim.workload.max_temperature_K == 600.0 and sim.schedule.step == 2
    sim.run()
    assert sim.schedule.finished


def test_a_yaml_config_reads_as_its_toml(tmp_path):
    # SPEC §10.5: protocols are TOML or YAML (safe_load only). The hybrid environment's Python 3.10 has
    # PyYAML but no tomli, so run.sh hands it the protocol as YAML (JSON is YAML).
    pytest.importorskip("yaml")
    path, _ = runner.protocol_file("swap_anneal_NaNbOCl4")
    data = run_command.parse_toml(path.read_text(encoding="utf-8"))
    for suffix, text in ((".yaml", json.dumps(data)), (".yml", "structure: a.cif\nrepeat: [3, 3, 3]\n")):
        config = tmp_path / f"protocol{suffix}"
        config.write_text(text, encoding="utf-8")
        assert run_command.parse_config(config.name, config.read_text(encoding="utf-8")) == \
            (data if suffix == ".yaml" else {"structure": "a.cif", "repeat": [3, 3, 3]})
    with pytest.raises(ValueError, match="mapping"):
        run_command.parse_config("x.yaml", "- 1\n- 2\n")
    with pytest.raises(ValueError):
        run_command.parse_config("x.yaml", "!!python/object:os.system {}\n")  # safe_load refuses tags
    assert run_command.parse_config("x.toml", "a = 1\n") == {"a": 1}
