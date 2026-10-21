"""Acceptance tools of plan task 2-T23, without a GPU.

- ``tools/ab_fastmace.py --release 1b``: this checkout (side 2) against the tag fastmace-1b in one process;
- ``tools/trajectory_versions.py``: default trajectories across versions, 1b-T14's measures;
- the phase 2 suites of ``tools/acceptance/suites.py`` and ``tools/acceptance/summarize2.py``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fastmace.bench import versions

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPOSITORY_ROOT / "tools"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(TOOLS))
ab_fastmace = _load(TOOLS / "ab_fastmace.py", "_tool_ab_fastmace_2t23")
trajectory_versions = _load(TOOLS / "trajectory_versions.py", "_tool_trajectory_versions_2t23")
sys.path.insert(0, str(TOOLS / "acceptance"))
import suites  # noqa: E402

summarize2 = _load(TOOLS / "acceptance" / "summarize2.py", "_tool_summarize2_2t23")


# --- tools/ab_fastmace.py --release 1b ---------------------------------------------------------------------


def _args(tmp_path, *extra):
    return ["--output", str(tmp_path / "out"), "--checkpoint", "m", "--renamed", str(tmp_path / "v"), *extra]


def test_release_1b_defaults_are_the_acceptance_protocol(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "1b"))
    assert args.sides == ["1b", "2"] and args.current == "2"
    assert args.expected_commit == "673a8a1" and args.blocks == 12 and args.block == 300 and args.warmup == 300
    assert args.model == "mh1-r2scan" and args.backend == "hybrid" and args.ensemble == "nvt"


def test_release_1a_is_unchanged(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path))
    assert args.sides == ["1a", "1b"] and args.current == "1b" and args.expected_commit == "bb35363"


def test_release_1b_sides(tmp_path, capsys):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "1b", "--sides", "2", "1b", "--pattern", "abba"))
    assert args.sides == ["2", "1b"]
    for sides in (("1a", "2"), ("2", "2"), ("1a",)):
        with pytest.raises(SystemExit):
            ab_fastmace.parse_args(_args(tmp_path, "--release", "1b", "--sides", *sides))
    with pytest.raises(SystemExit):  # side 2 belongs to --release 1b only
        ab_fastmace.parse_args(_args(tmp_path, "--sides", "2"))
    capsys.readouterr()


def test_release_1b_side_needs_the_renamed_copy(tmp_path, capsys):
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(["--output", str(tmp_path), "--checkpoint", "m", "--release", "1b"])
    assert "--name fastmace_1b" in capsys.readouterr().err
    args = ab_fastmace.parse_args(["--output", str(tmp_path), "--checkpoint", "m", "--release", "1b", "--sides", "2"])
    assert args.renamed is None


@pytest.mark.parametrize("extra", [("--backend", "cueq"), ("--ensemble", "npt"), ("--model", "polar1-s")])
def test_release_1b_refuses_what_one_process_cannot_hold(tmp_path, capsys, extra):
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(_args(tmp_path, "--release", "1b", *extra))
    assert "alternating processes" in capsys.readouterr().err
    assert ab_fastmace.parse_args(_args(tmp_path, "--release", "1b", *extra, "--sides", "2")).sides == ["2"]


def test_ratio_is_this_checkout_over_the_release_whatever_the_order():
    assert ab_fastmace.ratio_current_over_release({"1b": 10.0, "2": 10.1}, "2", "1b") == pytest.approx(1.01)
    assert ab_fastmace.ratio_current_over_release({"2": 10.1, "1b": 10.0}, "2", "1b") == pytest.approx(1.01)


def test_both_sides_of_release_1b_build_through_the_minimal_planner(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "1b"))
    assert ab_fastmace.Version("1b", args, None).minimal and ab_fastmace.Version("2", args, None).minimal
    old = ab_fastmace.parse_args(_args(tmp_path))
    assert not ab_fastmace.Version("1a", old, None).minimal and ab_fastmace.Version("1b", old, None).minimal


def test_the_renamed_1b_package_resolves_the_planner_modules(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "1b"))
    side = ab_fastmace.Version("2", args, sys.modules["fastmace"])
    assert side.module("planner.minimal").__name__ == "fastmace.planner.minimal"
    assert side.module("planner.workload").Workload is not None


def _tag_source(ref, path):
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(REPOSITORY_ROOT), "show", f"{ref}:{path}"],
                              check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def test_patch_targets_cover_every_patch_this_checkout_and_1b_install():
    attributes = {attribute for _, attribute in versions.PATCH_TARGETS}
    for source in ((REPOSITORY_ROOT / "fastmace" / "_patches.py").read_text(),
                   _tag_source("fastmace-1b", "fastmace/_patches.py")):
        if source is None:
            continue
        found = set(re.findall(r'_put\(\w+,\s*"(\w+)"', source))
        assert found and found <= attributes, found - attributes


# --- tools/trajectory_versions.py -----------------------------------------------------------------------


def test_trajectory_run_needs_the_renamed_copy_for_the_release(tmp_path, capsys):
    with pytest.raises(SystemExit):
        trajectory_versions.parse_args(["run", "--side", "1b", "--output", str(tmp_path), "--checkpoint", "m"])
    args = trajectory_versions.parse_args(["run", "--side", "2", "--output", str(tmp_path), "--checkpoint", "m"])
    assert args.steps == 20_000 and args.every == 20 and args.seeds == [20261001, 20261101]
    assert args.targets == [300.0, 600.0] and args.expected_commit == "673a8a1"
    capsys.readouterr()


def _rows(steps, every, energy, temperature, shift=0.0):
    return [{"step": s, "target": trajectory_versions.target(s, steps, (300.0, 600.0)), "energy": energy,
             "temperature": temperature, "positions": [[0.0, 0.0, shift + s * 1e-3]]}
            for s in range(every, steps + 1, every)]


def _write(directory, side, backend, runs, seeds):
    directory.mkdir()
    meta = dict(status="complete", side=side, fastmace_commit=side, backend=backend, steps=100, every=10,
                seeds=seeds, targets=[300.0, 600.0], structure="NaCl", model="mh1-r2scan",
                runs={str(seed): dict(plan=dict(options={"a": True}), rows=rows) for seed, rows in zip(seeds, runs)})
    (directory / "trajectory.json").write_text(json.dumps(meta))
    return directory


def test_compare_plain_cueq_positions(tmp_path):
    release = _write(tmp_path / "a", "1b", "cueq", [_rows(100, 10, -1.0, 300.0)], [1])
    same = _write(tmp_path / "b", "2", "cueq", [_rows(100, 10, -1.0, 300.0, 5e-13)], [1])
    off = _write(tmp_path / "c", "2", "cueq", [_rows(100, 10, -1.0, 300.0, 5e-12)], [1])
    result = trajectory_versions.compare(release, same, repeat=off)
    assert result["passed"] and result["positions_max_A"]["1"] == pytest.approx(5e-13)
    assert result["repeat_positions_max_A"]["1"] == pytest.approx(4.5e-12)
    assert not trajectory_versions.compare(release, off)["passed"]


def test_compare_hybrid_statistics(tmp_path):
    # Seeds differ by 2 K and 0.2 meV/atom; versions by 0.5 K and 0.01 meV/atom: within 2x the seeds.
    release = _write(tmp_path / "a", "1b", "hybrid", [_rows(100, 10, -1.0, 300.0), _rows(100, 10, -1.0002, 302.0)],
                     [1, 2])
    close = _write(tmp_path / "b", "2", "hybrid", [_rows(100, 10, -1.00001, 300.5), _rows(100, 10, -1.0002, 302.0)],
                   [1, 2])
    far = _write(tmp_path / "c", "2", "hybrid", [_rows(100, 10, -1.002, 310.0), _rows(100, 10, -1.0002, 302.0)],
                 [1, 2])
    result = trajectory_versions.compare(release, close)
    assert result["passed"] and result["limits"]["temperature_K"] == pytest.approx(4.0)
    assert not trajectory_versions.compare(release, far)["passed"]


def test_compare_refuses_different_protocols(tmp_path):
    release = _write(tmp_path / "a", "1b", "cueq", [_rows(100, 10, -1.0, 300.0)], [1])
    other = _write(tmp_path / "b", "2", "hybrid", [_rows(100, 10, -1.0, 300.0)], [1])
    with pytest.raises(ValueError, match="backend"):
        trajectory_versions.compare(release, other)


# --- suites and summary ---------------------------------------------------------------------------------


def _argv(job):
    return list(job.argv)


def test_regression2_suite():
    machine = suites.MACHINES["murphy"]
    jobs = suites.regression2_jobs(machine, Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    assert [job.name for job in jobs] == ["regression2-ab-abab", "regression2-ab-2-first", "regression2-ab-abba",
                                          "regression2-control-1b", "regression2-control-2"]
    for job in jobs:
        argv = _argv(job)
        assert job.timing and argv[argv.index("--release") + 1] == "1b"
        assert argv[argv.index("--structures") + 1:argv.index("--structures") + 3] == ["NaCl", "NbCl5"]
        assert ("--renamed" in argv) == ("1b" in argv[argv.index("--sides") + 1:argv.index("--warmup")])
    assert _argv(jobs[1])[_argv(jobs[1]).index("--sides") + 1:][:2] == ["2", "1b"]
    assert _argv(jobs[2])[_argv(jobs[2]).index("--pattern") + 1] == "abba"
    assert {_argv(job)[_argv(job).index("--blocks") + 1] for job in jobs[:3]} == {"12"}


def test_trajectory2_suite():
    machine = suites.MACHINES["local"]
    jobs = {job.name: job for job in suites.trajectory2_jobs(machine, Path("/home/u/fastmace-acceptance/r"))}
    assert set(jobs) == {"trajectory2-cueq-1b", "trajectory2-cueq-2", "trajectory2-cueq-2-repeat",
                         "trajectory2-hybrid-1b", "trajectory2-hybrid-2"}
    cueq = _argv(jobs["trajectory2-cueq-1b"])
    assert "mace_cueq_latest" in cueq[0] and "--positions" in cueq and "--renamed" in cueq
    hybrid = _argv(jobs["trajectory2-hybrid-2"])
    assert "mace_cueq_oeq" in hybrid[0] and hybrid[hybrid.index("--steps") + 1] == "20000"
    assert "--renamed" not in hybrid and not any(job.timing for job in jobs.values())


def test_tier_suites():
    machine = suites.MACHINES["murphy"]
    (acceptance,) = suites.acceptance_tier_jobs(machine, Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    (full,) = suites.full_tier_jobs(machine, Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    assert acceptance.timing and not full.timing  # the acceptance tier holds the step-time assertions
    def marker(job):  # the -m after "python -m pytest"
        argv = _argv(job)
        return argv[argv.index("-m", argv.index("pytest")) + 1]

    assert marker(acceptance) == "acceptance" and marker(full) == "not acceptance"
    assert {"FASTMACE_HYBRID_PYTHON", "FASTMACE_BASELINE"} <= set(acceptance.env)
    assert "FASTMACE_MODELS" in suites.environment(machine, Path("/mnt/homes/zwingel/fastmace-acceptance/r"))


def test_noise_floors_per_machine():
    assert summarize2.noise_floor("murphy", "NaCl") == summarize2.noise_floor("murphy", "NbCl5") == 0.0016
    assert summarize2.noise_floor("local", "NaCl") == 0.011 and summarize2.noise_floor("local", "NbCl5") == 0.004


def _ab_meta(ratio, valid=True, equal=True):
    blocks = [10.0] * 12
    return dict(current_side="2", release_side="1b", pattern="abab", build_order=["1b", "2"], sides=["1b", "2"],
                structures={"NaCl": dict(natoms=216, ratio=ratio, valid=valid, plan_check={"equal": equal},
                                         plan_describe_equal=equal,
                                         sides={"1b": dict(block_ms=blocks, median_ms=10.0),
                                                "2": dict(block_ms=[b * ratio for b in blocks],
                                                          median_ms=10.0 * ratio)})})


@pytest.mark.parametrize("machine, ratio, valid, equal, passed", [
    ("murphy", 1.001, True, True, True), ("murphy", 1.003, True, True, False), ("local", 1.009, True, True, True),
    ("murphy", 1.0, False, True, False)])
def test_regression_rows(machine, ratio, valid, equal, passed):
    (row,) = summarize2.regression_rows("p", _ab_meta(ratio, valid, equal), machine)
    assert row["passed"] is passed and row["ratio"] == ratio


def test_junit_totals(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text('<testsuites><testsuite tests="4" failures="1" errors="0" skipped="1" time="60">'
                    '<testcase classname="tests.gpu.test_a" name="test_x"/>'
                    '<testcase classname="tests.gpu.test_a" name="test_y"><failure/></testcase>'
                    '<testcase classname="tests.gpu.test_a" name="test_z"><skipped/></testcase>'
                    '<testcase classname="tests.gpu.test_b" name="test_w"/></testsuite></testsuites>')
    totals = summarize2.junit_totals(path)
    assert (totals["passed"], totals["failures"], totals["skipped"]) == (2, 1, 1)
    assert totals["failed_ids"] == ["tests.gpu.test_a::test_y"]


def test_run_takes_the_versions_directory(tmp_path, capsys):
    run = _load(TOOLS / "acceptance" / "run.py", "_tool_run_2t23")
    run.main(["regression2", "--machine", "murphy", "--run-id", "x", "--fastmace", "/mnt/homes/zwingel/f-2-T23/src",
              "--versions", "/mnt/homes/zwingel/f-2-T23/versions", "--dry-run"])
    out = capsys.readouterr().out
    assert "--renamed /mnt/homes/zwingel/f-2-T23/versions" in out and "/mnt/homes/zwingel/f-2-T23/src/tools" in out


# --- tools/runner_statistics.py ---------------------------------------------------------------------------

runner_statistics = _load(TOOLS / "runner_statistics.py", "_tool_runner_statistics_2t23")


def _runner_case(directory, case, rows_by_seed):
    directory.mkdir()
    seeds = list(rows_by_seed)
    meta = dict(status="complete", seeds=seeds, runs={str(seed): dict(runner=case, backend="hybrid") for seed in seeds})
    (directory / "statistics.json").write_text(json.dumps(meta))
    return directory


def test_runner_statistics_compare(tmp_path, monkeypatch):
    rows = {("f", "1"): (300.0, -1.0), ("f", "2"): (302.0, -1.0002), ("r", "1"): (300.5, -1.00001),
            ("r", "2"): (302.0, -1.0002)}

    def fake_rows(directory):
        kind = "f" if directory.parent.name == "fresh" else "r"
        temperature, energy = rows[(kind, directory.name.split("-")[1])]
        return [{"step": s, "target": 300.0 + s, "temperature": temperature, "energy": energy}
                for s in range(20, 301, 20)]

    monkeypatch.setattr(runner_statistics, "thermo_rows", fake_rows)
    fresh = _runner_case(tmp_path / "fresh", "in-process", {1: None, 2: None})
    runner = _runner_case(tmp_path / "runner", "subprocess", {1: None, 2: None})
    result = runner_statistics.compare(fresh, runner)
    assert result["passed"] and result["limits"]["temperature_K"] == pytest.approx(4.0)
    rows[("r", "1")] = (310.0, -1.0)
    assert not runner_statistics.compare(fresh, runner)["passed"]
    rows[("r", "1")] = (300.5, -1.00001)
    in_process = _runner_case(tmp_path / "runner2", "in-process", {1: None, 2: None})
    monkeypatch.setattr(runner_statistics, "thermo_rows",
                        lambda d: fake_rows(d.parent.with_name("runner") / d.name))
    assert not runner_statistics.compare(fresh, in_process)["passed"]  # the runner must really have run


def test_runner_statistics_reads_simulation_thermo_columns(tmp_path):
    from fastmace.simulation import THERMO_HEADER
    assert {"step", "target_K", "temperature_K", "potential_eV_atom"} <= set(THERMO_HEADER)


def test_runner_statistics_suite():
    jobs = suites.runner_statistics_jobs(suites.MACHINES["murphy"], Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    assert [job.name for job in jobs] == ["runner-statistics-fresh", "runner-statistics-runner"]
    assert all("mace_cueq_oeq" in job.argv[0] and not job.timing for job in jobs)


def test_runner_statistics_four_seeds_use_the_median_pair(tmp_path, monkeypatch):
    temperatures = {"f": {"1": 300.0, "2": 301.0, "3": 302.0, "4": 310.0}, "r": {"1": 300.5, "2": 301.0, "3": 302.0,
                                                                              "4": 310.0}}

    def fake_rows(directory):
        kind = "f" if directory.parent.name == "fresh" else "r"
        temperature = temperatures[kind][directory.name.split("-")[1]]
        return [{"step": s, "target": 300.0 + s, "temperature": temperature, "energy": -1.0}
                for s in range(20, 301, 20)]

    monkeypatch.setattr(runner_statistics, "thermo_rows", fake_rows)
    seeds = {1: None, 2: None, 3: None, 4: None}
    fresh = _runner_case(tmp_path / "fresh", "in-process", seeds)
    runner = _runner_case(tmp_path / "runner", "subprocess", seeds)
    result = runner_statistics.compare(fresh, runner)
    # fresh pairs: 1, 2, 10, 1, 9, 8 K -> median 5 K -> limit 10 K
    assert result["between_seeds"]["temperature_K"] == pytest.approx(5.0)
    assert result["limits"]["temperature_K"] == pytest.approx(10.0) and result["passed"]


def test_runner_statistics_refuses_different_start_states(tmp_path):
    fresh = _runner_case(tmp_path / "fresh", "in-process", {1: None, 2: None})
    runner = _runner_case(tmp_path / "runner", "subprocess", {1: None, 2: None})
    meta = json.loads((runner / "statistics.json").read_text())
    meta["start"] = "/elsewhere/start.extxyz"
    (runner / "statistics.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="start"):
        runner_statistics.compare(fresh, runner)


def test_runner_statistics_equilibrate_arguments(tmp_path, capsys):
    args = runner_statistics.parse_args(["equilibrate", "--output", str(tmp_path / "e")])
    assert args.stages == [0.5, 4000, 2.0, 5000] and args.temperature_K == 300.0
    with pytest.raises(SystemExit):
        runner_statistics.parse_args(["equilibrate", "--output", str(tmp_path / "e"), "--stages", "0.5"])
    capsys.readouterr()


def test_runner_statistics_reads_a_start_with_momenta(tmp_path):
    from ase.build import bulk
    from ase.io import write
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    path = tmp_path / "start.extxyz"
    write(path, atoms)
    with pytest.raises(ValueError, match="momenta"):
        runner_statistics.read_start(path)
    atoms.set_momenta([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]])
    write(path, atoms)
    assert runner_statistics.read_start(path).get_momenta()[0, 0] == pytest.approx(0.1)


def test_runner_repeat_suite():
    root = Path("/mnt/homes/zwingel/fastmace-acceptance/r")
    jobs = suites.runner_statistics_repeat_jobs(suites.MACHINES["murphy"], root)
    assert [job.name for job in jobs] == ["runner-repeat-equilibrate", "runner-repeat-fresh", "runner-repeat-runner"]
    start = "/mnt/homes/zwingel/fastmace-acceptance/r/runner-repeat-equilibrate/start.extxyz"
    for job in jobs[1:]:
        argv = list(job.argv)
        assert argv[argv.index("--start") + 1] == start
        assert argv[argv.index("--seeds") + 1:] == ["20261001", "20261101", "20261201", "20261301"]
    assert not any(job.timing for job in jobs)


def test_regression2_nacl_suite():
    jobs = suites.regression2_nacl_jobs(suites.MACHINES["murphy"], Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    orders = []
    for job in jobs:
        argv = list(job.argv)
        assert job.timing and argv[argv.index("--structures") + 1:argv.index("--sides")] == ["NaCl"]
        orders.append((argv[argv.index("--sides") + 1], argv[argv.index("--pattern") + 1]))
    assert orders == [("1b", "abab"), ("2", "abab"), ("1b", "abba")]


def test_tier_suite_sets_the_polar_references_only_where_they_exist(tmp_path):
    import dataclasses
    machine = dataclasses.replace(suites.MACHINES["local"], root=str(tmp_path))
    (job,) = suites.full_tier_jobs(machine, tmp_path / "run")
    assert "FASTMACE_1A_T19_POLAR1S" not in job.env
    (tmp_path / "1a-T19-gate-leto-20261018" / "gate-polar1-s").mkdir(parents=True)
    (job,) = suites.full_tier_jobs(machine, tmp_path / "run")
    assert job.env["FASTMACE_1A_T19_POLAR1S"].endswith("gate-polar1-s")


# --- the combined acceptance (optimisation round 1) -------------------------------------------------------


def test_compare_options_allows_the_expected_changes():
    release = {"block_linear": True}
    current = {"block_linear": True, "gated_rows": True, "off:segment_gemm": "fused only"}
    check = ab_fastmace.compare_options(release, current, allowed=("gated_rows", "segment_gemm"))
    assert not check["equal"] and check["as_expected"] and check["unexpected"] == []
    check = ab_fastmace.compare_options(release, {**current, "block_linear": None}, allowed=("gated_rows",))
    assert set(check["unexpected"]) == {"block_linear", "off:segment_gemm"} and not check["as_expected"]


def test_regression2_expects_the_round_changes_only():
    (job, *_) = suites.regression2_jobs(suites.MACHINES["murphy"], Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    argv = list(job.argv)
    at = argv.index("--allow-option-changes")
    assert argv[at + 1:at + 3] == ["gated_rows", "segment_gemm"]
    args = ab_fastmace.parse_args(argv[2:])
    assert args.allow_option_changes == ["gated_rows", "segment_gemm"]


def test_trajectory_compare_with_expected_option_changes(tmp_path):
    release = _write(tmp_path / "a", "1b", "cueq", [_rows(100, 10, -1.0, 300.0)], [1])
    current = _write(tmp_path / "b", "2", "cueq", [_rows(100, 10, -1.0, 300.0, 5e-13)], [1])
    meta = json.loads((current / "trajectory.json").read_text())
    meta["runs"]["1"]["plan"]["options"]["gated_rows"] = True
    (current / "trajectory.json").write_text(json.dumps(meta))
    assert not trajectory_versions.compare(release, current)["passed"]
    assert trajectory_versions.compare(release, current, allowed=("gated_rows",))["passed"]


def test_gate_split_covers_the_19_labels_once():
    assert sorted(suites.LOCAL_GATE_LABELS + suites.MURPHY_GATE_LABELS) == sorted(suites.GATE_LABELS)
    assert len(suites.GATE_LABELS) == 19 and not set(suites.LOCAL_GATE_LABELS) & set(suites.MURPHY_GATE_LABELS)
    root = Path("/home/u/fastmace-acceptance/r")
    local = {job.name for job in suites.gate_local_jobs(suites.MACHINES["local"], root)}
    assert local and all(name.startswith("gate-") for name in local)


def test_round_tests_job_selects_the_round_files_outside_the_acceptance_tier():
    import dataclasses
    machine = dataclasses.replace(suites.MACHINES["local"], fastmace=str(REPOSITORY_ROOT))
    (job,) = suites.round_tests_jobs(machine, Path("/home/u/fastmace-acceptance/r"))
    argv = list(job.argv)
    assert argv[argv.index("-m", argv.index("pytest")) + 1] == "not acceptance" and not job.timing
    files = [Path(item).name for item in argv if item.endswith(".py")]
    assert "test_canonical_restart_gpu.py" in files and "test_reproducible_gpu.py" in files


@pytest.mark.parametrize("ratio, one_sided, passed", [
    (0.96, True, True), (0.96, False, False), (1.0015, True, True), (1.0017, True, False)])
def test_regression_rows_one_sided(ratio, one_sided, passed):
    (row,) = summarize2.regression_rows("p", _ab_meta(ratio), "murphy", one_sided=one_sided)
    assert row["passed"] is passed and row["one_sided"] is one_sided


def test_gate_omol0_suite_runs_frozen_and_without_the_template_cache():
    jobs = suites.gate_omol0_jobs(suites.MACHINES["murphy"], Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    assert [job.name for job in jobs] == ["gate-omol0-xl", "gate-omol0-xl-no-template-cache"]
    frozen, off = (list(job.argv) for job in jobs)
    assert "--engine-config" not in frozen
    assert off[off.index("--engine-config") + 1] == "model_template_cache=false"
    assert off[off.index("--output") + 1].endswith("gate-omol0-xl-no-template-cache")


def test_gate_tool_engine_overrides():
    gate = _load(TOOLS / "force_stress_gate.py", "_tool_force_stress_gate_2t23")
    assert gate.engine_overrides(["model_template_cache=false"]) == {"model_template_cache": False}
    for bad in (["backend=cueq"], ["model_template_cache=off"], ["no_such_field=true"]):
        with pytest.raises(ValueError):
            gate.engine_overrides(bad)


def test_simulation_tests_job_covers_every_tier_of_the_simulation_files():
    (job,) = suites.simulation_tests_jobs(suites.MACHINES["murphy"], Path("/mnt/homes/zwingel/fastmace-acceptance/r"))
    argv = list(job.argv)
    assert argv[argv.index("-m", argv.index("pytest")) + 1] == "acceptance or not acceptance" and not job.timing
    files = {Path(item).name for item in argv if item.endswith(".py")}
    assert files == set(suites.SIMULATION_TEST_FILES)
    for name in files:
        assert (REPOSITORY_ROOT / "tests" / "gpu" / name).is_file()


# --- tools/ab_fastmace.py --release 2 (native Ada A/B on leeloo, 22 October 2026) ----------------------------


def test_release_2_sides_and_commit(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "2"))
    assert args.sides == ["2", "3"] and args.current == "3" and args.expected_commit == "e52c482"
    assert ab_fastmace.RELEASES["2"].tag == "fastmace-2" and ab_fastmace.RELEASES["2"].renamed == "fastmace_2"


def test_engine_options_apply_to_this_checkout_only(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "2", "--engine-option", "fused_integrator=on"))
    assert args.engine_options == {"fused_integrator": True}
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(_args(tmp_path, "--release", "2", "--engine-option", "fused_integrator"))
