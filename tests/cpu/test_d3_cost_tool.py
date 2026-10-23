"""tools/bench/d3_cost.py without a GPU: the route, its thermostat targets and the per-segment summary."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.fixture(scope="module")
def tool():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("_d3_cost", TOOLS / "bench" / "d3_cost.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def test_anneal_route_follows_the_profile(tool):
    route = tool.parse_route(tool.ANNEAL_ROUTE)
    assert [s.label for s in route] == ["heat", "peak", "transit", "cool600", "transit", "cool300"]
    assert route[0].steps == 20000 and route[0].end_K == 1000.0  # 50 K/ps at 1 fs
    for segment in route:
        if segment.label in ("peak", "cool600", "cool300"):  # 3 K/ps at 1 fs
            assert (segment.start_K - segment.end_K) / segment.steps == pytest.approx(3e-3)
    for previous, segment in zip(route, route[1:]):
        assert previous.end_K == segment.start_K


def test_route_target(tool):
    route = tool.parse_route("a:0:100:10,b:100:50:5")
    assert tool.route_target(route, 0) == 0.0
    assert tool.route_target(route, 5) == 50.0
    assert tool.route_target(route, 10) == 100.0
    assert tool.route_target(route, 12) == 80.0
    assert tool.route_target(route, 99) == 50.0


def test_segment_summary(tool):
    def block(side, pair, ms, builds):
        return dict(segment="s", side=side, pair=pair, steps=100, ms_per_step=ms, volume_A3=1.0, recompiled=False,
                    delta=dict(neighbour_rebuilds=1, recaptures=0, d3_builds=builds, d3_term_build_s=0.01 * builds))
    blocks = [block("off", 0, 10.0, 0), block("on", 0, 11.0, 2), block("on", 1, 11.5, 1), block("off", 1, 10.0, 0)]
    out = tool.segment_summary(blocks, "s")
    assert out["off"]["mean_ms"] == pytest.approx(10.0)
    assert out["on"]["mean_ms"] == pytest.approx(11.25)
    assert out["overhead_mean"] == pytest.approx(0.125)
    assert out["overhead_pairs"] == pytest.approx([0.1, 0.15])
    assert out["on"]["d3_builds"] == 3
    assert out["d3_build_ms_per_step"] == pytest.approx(1e3 * 0.03 / 200)
