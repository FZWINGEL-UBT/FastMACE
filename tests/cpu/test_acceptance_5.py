"""Phase 5's acceptance tools (5-T11): ``tools/acceptance/phase5_local.sh`` parses and has a case for every job of
the runbook; the release side against phase 4 (``tools/ab_fastmace.py --release 4``,
``tools/trajectory_versions.py --against 4``) takes sides 4 and 5; ``tools/validation/adapt_runs.py check`` fails a
run that switched tiers and records the re-capture fractions, and its NaCl anneal protocol loads."""

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "acceptance" / "phase5_local.sh"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
def test_the_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_every_job_of_the_runbook_has_a_case():
    text = SCRIPT.read_text(encoding="utf-8")
    cases = set()
    for line in re.findall(r"^\s{2}([a-z0-9|-]+)\)", text, re.M):
        cases.update(line.split("|"))
    assert {"conformance", "cpu-acceptance", "corpus", "trajectories-cpu", "trajectories-reproducible",
            "restart-across", "regression", "adapt-storm", "adapt-production", "calibration", "adapt-tests",
            "preflight", "targeted", "compat", "acceptance-tier", "slow-tier"} <= cases


@needs_bash
def test_jobs_refuse_without_their_baseline(tmp_path):
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "FASTMACE_REPOSITORY": str(tmp_path)}  # no tag here
    unknown = subprocess.run(["bash", str(SCRIPT), str(tmp_path), "no-such-job"], env=env, capture_output=True)
    assert unknown.returncode == 2
    for job in ("corpus", "regression", "trajectories-cpu", "restart-across"):
        baseline = subprocess.run(["bash", str(SCRIPT), str(tmp_path), job], env=env, capture_output=True)
        assert baseline.returncode == 2 and b"fastmace-4" in baseline.stderr, job


def _tool(name):
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(f"_tool_{name}_5t11", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_4_is_the_fastmace_4_tag_with_side_5(tmp_path):
    ab = _tool("ab_fastmace")
    release = ab.RELEASES["4"]
    assert (release.tag, release.commit, release.renamed, release.planner, release.current) == (
        "fastmace-4", "dae3848", "fastmace_4", "minimal", "5")
    common = ["--output", str(tmp_path / "out"), "--checkpoint", "m", "--renamed", str(tmp_path / "v")]
    args = ab.parse_args([*common, "--release", "4", "--expected-commit", "dae3848"])
    assert args.sides == ["4", "5"] and args.current == "5" and args.integrator_options == {}
    assert ab.parse_args([*common, "--release", "4"]).expected_commit == "dae3848"  # the tag's commit
    for sides in (("3b", "5"), ("4", "4")):
        with pytest.raises(SystemExit):
            ab.parse_args([*common, "--release", "4", "--expected-commit", "dae3848", "--sides", *sides])


def test_trajectory_runs_against_4(tmp_path):
    tv = _tool("trajectory_versions")
    common = ["run", "--output", str(tmp_path / "out"), "--checkpoint", "m"]
    assert tv.parse_args([*common, "--against", "4", "--side", "5", "--device", "cpu",
                          "--dispersion", "d3bj"]).dispersion == "d3bj"
    assert tv.parse_args([*common, "--against", "4", "--side", "4", "--renamed", "v",
                          "--expected-commit", "dae3848"]).side == "4"
    with pytest.raises(SystemExit):
        tv.parse_args([*common, "--against", "4", "--side", "4b"])
    assert tv.parse_args([*common, "--against", "4", "--side", "4", "--renamed", "v"]).expected_commit == "dae3848"


def test_adapt_runs_checks_switches_and_writes_the_nacl_protocol(tmp_path):
    import json

    spec = importlib.util.spec_from_file_location("_adapt_runs_5t11", ROOT / "tools" / "validation" / "adapt_runs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, switches in (("quiet", []), ("switched", [{"step": 50, "from": "graph", "to": "compiled"}])):
        (tmp_path / name).mkdir()
        segments = [{"wall_s": 10.0, "recapture_s": 0.5}, {"wall_s": 10.0, "recapture_s": 1.5}]
        (tmp_path / name / "metadata.json").write_text(json.dumps(
            {"counters": {"tier_switches": switches, "segments": segments, "recaptures": 3}}))
    assert module.main(["check", str(tmp_path / "quiet")]) == 0
    assert module.main(["check", str(tmp_path / "quiet"), str(tmp_path / "switched")]) == 1
    entry = module.check([tmp_path / "quiet"])[str(tmp_path / "quiet")]
    assert entry["recapture_fraction"] == pytest.approx(0.1) and entry["max_segment_recapture_fraction"] == 0.15
    # The protocol reads as fastmace run reads it: stages split off, the rest fm.md's arguments.
    from fastmace.cli.run import md_arguments, parse_config
    from fastmace.protocols import runner

    path = module.nacl_config(tmp_path / "nacl")
    stages, md_data = runner.split_stages(parse_config(str(path), path.read_text()))
    md_data["output"] = {**md_data["output"], "directory": str(tmp_path / "out")}  # fastmace run's --output
    structure, options = md_arguments(md_data)
    assert Path(structure).is_file() and options["ensemble"] == "NVT" and stages["relax"] is not None
