"""tools/ab_rewrites.py without a GPU: its two rewrite sets."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("_tool_ab_rewrites", TOOLS / "ab_rewrites.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load()


def test_sides_parse_and_must_differ():
    args = tool.parse_args(["--output", "o", "--checkpoint", "c", "--a", "gated_rows=on", "--b", "gated_rows=on",
                            "segment_gemm=on"])
    assert args.a == {"gated_rows": "on"} and args.b == {"gated_rows": "on", "segment_gemm": "on"}
    with pytest.raises(SystemExit):
        tool.parse_args(["--output", "o", "--checkpoint", "c", "--a", "gated_rows=on", "--b", "gated_rows=on"])
    with pytest.raises(SystemExit):
        tool.parse_args(["--output", "o", "--checkpoint", "c", "--b", "nonsense=on"])


def test_engine_sides_take_booleans_floats_and_choices():
    assert tool.engine_overrides(["step_graph=true", "force_graphs=keep"]) == {"step_graph": True,
                                                                               "force_graphs": "keep"}
    for bad in (["step_graph=on"], ["no_such_field=1"], ["force_graphs="]):
        with pytest.raises(ValueError):
            tool.engine_overrides(bad)
