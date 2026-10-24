"""``tools/restart_versions.py`` (4-T13): its arguments, the release root's commit check and the comparison."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tool():
    spec = importlib.util.spec_from_file_location("_restart_versions", ROOT / "tools" / "restart_versions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_release_stages_need_their_root_and_commit_and_resume_its_source():
    tool = _tool()
    common = ["--model", "m", "--checkpoint", "m", "--output", "o"]
    args = tool.parse_args(["run", "--stage", "write", "--release-root", "r", "--expected-commit", "ff80ca2", *common])
    assert (args.steps, args.restart_step, args.device) == (100, 50, "cpu")
    for argv in (["run", "--stage", "write", *common], ["run", "--stage", "resume", *common],
                 ["run", "--stage", "resume", "--source", "s", "--restart-step", "100", *common]):
        with pytest.raises(SystemExit):
            tool.parse_args(argv)


def test_the_release_root_must_be_the_expected_commit(tmp_path):
    tool = _tool()
    (tmp_path / ".source_commit").write_text("ff80ca2a33d3c1ce25b3181095ff7b2a42a7f00d\n")
    assert tool.release_source(tmp_path, "ff80ca2").startswith("ff80ca2")
    with pytest.raises(SystemExit):
        tool.release_source(tmp_path, "abc1234")


def test_compare_is_bitwise_over_step_positions_momenta_and_cell(tmp_path):
    tool = _tool()
    final = {"step": 100, "positions": "00ff", "momenta": "0a0b", "cell": "0c", "energy": -1.0}
    for name, changes in (("a", {}), ("b", {}), ("c", {"momenta": "0a0c"})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "restart_versions.json").write_text(json.dumps({"final": {**final, **changes}}))
    assert tool.compare(tmp_path / "a", tmp_path / "b")["bitwise"]
    result = tool.compare(tmp_path / "a", tmp_path / "c")
    assert not result["bitwise"] and result["differs"] == ["momenta"]
