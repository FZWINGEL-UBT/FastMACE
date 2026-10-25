"""Phase 6 acceptance tools (6-T13): release ``5`` (the ``fastmace-5`` tag) with side ``6``."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"_tool_{name}_6t13", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_5_is_the_fastmace_5_tag_with_side_6(tmp_path):
    ab = _tool("ab_fastmace")
    release = ab.RELEASES["5"]
    assert (release.tag, release.commit, release.renamed, release.planner, release.current) == (
        "fastmace-5", "6722f29", "fastmace_5", "minimal", "6")
    common = ["--output", str(tmp_path / "out"), "--checkpoint", "m", "--renamed", str(tmp_path / "v")]
    args = ab.parse_args([*common, "--release", "5"])
    assert args.sides == ["5", "6"] and args.current == "6" and args.expected_commit == "6722f29"


def test_trajectory_runs_against_5(tmp_path):
    tv = _tool("trajectory_versions")
    common = ["run", "--output", str(tmp_path / "out"), "--checkpoint", "m"]
    assert tv.parse_args([*common, "--against", "5", "--side", "6", "--device", "cpu"]).side == "6"
    assert tv.parse_args([*common, "--against", "5", "--side", "5", "--renamed", "v"]).expected_commit == "6722f29"
    with pytest.raises(SystemExit):
        tv.parse_args([*common, "--against", "5", "--side", "4"])
