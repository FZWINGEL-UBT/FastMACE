"""tools/force_stress_gate.py --rewrites (OPT-T02): RewriteConfig fields for FastMACE's minimal planner."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("_tool_gate_rewrites", TOOLS / "force_stress_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load()
BASE = ["--output", "out", "--checkpoint", "mh1.model", "--structures-root", "."]


def test_overrides_parse_toggles_and_bools():
    assert gate.rewrite_overrides(["gated_rows=on", "polar_uvu=false"]) == {"gated_rows": "on", "polar_uvu": False}
    assert gate.rewrite_overrides([]) == {}
    for bad in (["nonsense=on"], ["gated_rows"], ["polar_uvu=maybe"]):
        with pytest.raises(ValueError):
            gate.rewrite_overrides(bad)


def test_the_option_belongs_to_fastmace_s_minimal_planner():
    args = gate.parse_args([*BASE, "--engine", "fastmace", "--rewrites", "gated_rows=on"])
    assert args.rewrites == {"gated_rows": "on"}
    with pytest.raises(SystemExit):
        gate.parse_args([*BASE, "--engine", "fastmace", "--planner", "fixed", "--rewrites", "gated_rows=on"])
    with pytest.raises(SystemExit):
        gate.parse_args([*BASE, "--baseline", ".", "--rewrites", "gated_rows=on"])
