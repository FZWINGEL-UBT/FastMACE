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


def test_the_gate_applies_rule_13_as_production_does():
    """A self-check failure at the build revises the plan (rule 13) and builds again, as production's fallback
    does; before, the gate tool's direct ``planner.build`` ended the run (OMOL-0 on the A100, OPT4-T03)."""
    from types import SimpleNamespace

    import test_fallback_candidates as tfc

    from fastmace.errors import SelfCheckError

    _, plan = tfc.gpu_plan()
    failed = tfc.enabled_fields(plan)[0]
    error = SelfCheckError("self-check failed: 1.3e-4 > 1e-4")
    error.reaction = SimpleNamespace(rewrite=failed.upper(), disabled=(failed.upper(),), fields=(failed,),
                                     reason="1.3e-4 > 1e-4")
    built = []

    def build(candidate):
        built.append(candidate)
        if getattr(candidate.rewrites, failed) not in ("off", False):
            raise error
        return "engine"

    engine, final, revisions = gate.build_with_rule13(build, plan)
    assert engine == "engine" and len(built) == 2 and final.predecessor is plan
    assert getattr(final.rewrites, failed) in ("off", False) and revisions == [final.reason]
    assert revisions[0].startswith("rule 13:")
    assert gate.build_with_rule13(lambda candidate: "first", plan) == ("first", plan, [])
