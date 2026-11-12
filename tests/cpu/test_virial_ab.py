"""tools/virial_ab.py's module-constant sides (12 November 2026, the stress-mode A/B of LATE_PULLBACK): parsing, and the
proof counters read as ints or functions."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fastmace_tools_virial_ab", ROOT / "tools" / "virial_ab.py")
virial_ab = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = virial_ab
spec.loader.exec_module(virial_ab)


def test_parse_constant():
    w = "fastmace.kernels.oeq_streaming.wprime_backward"
    assert virial_ab.parse_constant(f"{w}:LATE_PULLBACK=True") == (w, "LATE_PULLBACK", True)
    assert virial_ab.parse_constant(f"{w}:CHANNEL_VECTOR=2") == (w, "CHANNEL_VECTOR", 2)
    assert virial_ab.parse_constant("m:N=0.5") == ("m", "N", 0.5)
    assert virial_ab.parse_constant("m:N=None") == ("m", "N", None)
    assert virial_ab.parse_constant("m:N=float4") == ("m", "N", "float4")
    for bad in ("m:N", "N=1", ":N=1", "m:=1"):
        with pytest.raises(ValueError):
            virial_ab.parse_constant(bad)


def test_counters_read_ints_and_functions():
    from fastmace.kernels.oeq_streaming import wprime_pullback as wpb
    item = "fastmace.kernels.oeq_streaming.wprime_pullback:APPLIED"
    assert virial_ab.counters([item]) == {item: wpb.APPLIED()}
    from fastmace.kernels.oeq_streaming import wprime_backward as wp
    plain = "fastmace.kernels.oeq_streaming.wprime_backward:LATE_PULLBACK_GPUS"  # not callable: read as it is
    assert virial_ab.counters([plain]) == {plain: wp.LATE_PULLBACK_GPUS}
