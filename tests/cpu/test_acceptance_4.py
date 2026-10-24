"""``tools/acceptance/phase4_local.sh`` (4-T13): it parses, its preflight files exist, and every job it lists is a
case it handles."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "acceptance" / "phase4_local.sh"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
def test_the_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_the_preflight_files_exist():
    text = SCRIPT.read_text(encoding="utf-8")
    block = re.search(r"PREFLIGHT_FILES=\((.*?)\)", text, re.S).group(1)
    files = block.split()
    assert files and all((ROOT / name).is_file() for name in files), [n for n in files if not (ROOT / n).is_file()]


def test_every_job_of_the_runbook_has_a_case():
    text = SCRIPT.read_text(encoding="utf-8")
    cases = set()
    for line in re.findall(r"^\s{2}([a-z0-9|-]+)\)", text, re.M):
        cases.update(line.split("|"))
    assert {"regression", "regression-npt", "ase-speed", "preflight", "optimisers", "minima", "protocol-start",
            "protocol-resume", "nve-drift", "trajectories-cpu", "trajectories-reproducible", "restart-across",
            "cpu-acceptance", "acceptance-tier", "slow-tier", "targeted", "compat"} <= cases


# --- tools/ab_fastmace.py --release 3b, tools/trajectory_versions.py --against 3b --dispersion ----------------


def _tool(name):
    import importlib.util
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(f"_tool_{name}_4t13", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_3b_is_the_fastmace_3b_tag_with_side_4(tmp_path):
    ab = _tool("ab_fastmace")
    release = ab.RELEASES["3b"]
    assert (release.tag, release.commit, release.renamed, release.planner, release.current) == (
        "fastmace-3b", None, "fastmace_3b", "minimal", "4")
    common = ["--output", str(tmp_path / "out"), "--checkpoint", "m", "--renamed", str(tmp_path / "v")]
    args = ab.parse_args([*common, "--release", "3b", "--expected-commit", "ff80ca2"])
    assert args.sides == ["3b", "4"] and args.current == "4" and args.integrator_options == {}
    with pytest.raises(SystemExit):  # the release side needs its commit while the tag is not set
        ab.parse_args([*common, "--release", "3b"])
    for sides in (("3", "4"), ("3b", "3b")):
        with pytest.raises(SystemExit):
            ab.parse_args([*common, "--release", "3b", "--expected-commit", "ff80ca2", "--sides", *sides])
    npt = ab.parse_args([*common, "--release", "3b", "--expected-commit", "ff80ca2", "--ensemble", "npt",
                         "--sides", "4"])
    assert npt.sides == ["4"]


def test_trajectory_runs_against_3b_with_and_without_d3(tmp_path):
    tv = _tool("trajectory_versions")
    common = ["run", "--output", str(tmp_path / "out"), "--checkpoint", "m"]
    args = tv.parse_args([*common, "--against", "3b", "--side", "4", "--device", "cpu", "--dispersion", "d3bj"])
    assert args.dispersion == "d3bj"
    assert tv.parse_args([*common, "--against", "3b", "--side", "3b", "--renamed", "v",
                          "--expected-commit", "ff80ca2"]).dispersion == "none"


def test_compare_refuses_runs_with_different_dispersion(tmp_path):
    import json

    tv = _tool("trajectory_versions")
    meta = {"backend": "auto", "steps": 10, "every": 10, "seeds": [1], "targets": [300.0, 300.0],
            "structure": "NaCl", "model": "m", "status": "complete", "runs": {"1": {"rows": []}}}
    for name, dispersion in (("a", "none"), ("b", "d3bj")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "trajectory.json").write_text(json.dumps({**meta, "dispersion": dispersion}))
    with pytest.raises(ValueError, match="dispersion"):
        tv.compare(tmp_path / "a", tmp_path / "b")
