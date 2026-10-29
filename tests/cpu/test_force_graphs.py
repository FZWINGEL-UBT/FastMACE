"""Force-graph variants near the memory limit (plans/capacity.md, CAP-T01; ``EngineConfig.force_graphs``).

A force graph (B, ACB, ACPB) keeps one set of activations in the graph pool. A capture at a larger padded edge size
beside it needs a second set, and on the 16 GB card that capture was the first allocation to fail
(docs/validation/capacity-16gb.md). ``"one"`` releases the captured graphs and their pool before a new force
variant is captured, so one force graph lives at a time; ``"fit"`` does that only when the new variant would not fit
the free device memory, estimated from the bytes per edge the earlier force captures added; ``"keep"`` is the
earlier behaviour. After a release the padded edge size never drops below the captured one (padding is exact), so
a count that falls back does not recapture. On the CPU the replay engine stands in for CUDA graphs and the free
memory is injected.
"""

import dataclasses

import numpy as np
import pytest
import torch
from test_engine import build

from fastmace.config import EngineConfig
from fastmace.engine import graph

STEPS = 30


def growing_engine(monkeypatch, force_graphs, **engine):
    """A checked replay engine whose padded edge size grows by one every step, then falls back by two every
    fifth step: every step needs a force variant, and some ask for a smaller size than one captured before.
    Padding is exact (the argon cell's count stays put at these temperatures)."""
    config = EngineConfig(pipelined="off", async_output="off", compile_mode="none", device="cpu",
                          force_graphs=force_graphs, **engine)
    out = build(engine=config, replay=True)

    def padded(count, capacity, bucket):
        step = int(out.state.step)
        return min(capacity, count + step - 2 * (step % 5 == 4))

    monkeypatch.setattr(graph, "round_up_edges", padded)
    return out


def force_keys(engine):
    return [key for key in engine._graphs if graph._kind(key) in graph.FORCE_KINDS]


def heat(engine, steps=STEPS):
    held, sizes = [], []
    for step in range(1, steps + 1):
        engine.integrator.set_temperature(min(3.0 * step, 150.0))
        engine.step()
        held.append(len(force_keys(engine)))
        sizes.append(engine.edge_sizes_used and max(engine.edge_sizes_used))
    return held, sizes


@pytest.mark.filterwarnings("ignore::fastmace.engine.graph.RecaptureBudgetWarning")
def test_keep_holds_every_force_variant(monkeypatch):
    engine = growing_engine(monkeypatch, "keep")
    held, _ = heat(engine)
    assert max(held) > 2
    assert engine.info()["force_graph_releases"] == 0


@pytest.mark.filterwarnings("ignore::fastmace.engine.graph.RecaptureBudgetWarning")
def test_one_holds_one_force_graph_and_never_pads_below_the_captured_size(monkeypatch):
    engine = growing_engine(monkeypatch, "one")
    held, _ = heat(engine)
    assert max(held) == 1
    info = engine.info()
    assert info["force_graph_releases"] > 0
    released = [entry for entry in info["invalidations"] if entry["reason"].startswith("one force graph")]
    assert len(released) == info["force_graph_releases"]
    # The sizes asked for fall back every fifth step; the floor keeps the larger captured size, so those
    # steps replay it instead of capturing a smaller one.
    captured = [int(entry["key"].split(", ")[1].rstrip(")")) for entry in info["recapture_log"]
                if entry["key"].startswith("('B'")]
    assert captured == sorted(captured)
    assert info["force_graph_floor"] == captured[-1]


@pytest.mark.filterwarnings("ignore::fastmace.engine.graph.RecaptureBudgetWarning")
def test_one_and_keep_give_the_same_trajectory(monkeypatch):
    runs = []
    for mode in ("keep", "one"):
        engine = growing_engine(monkeypatch, mode)
        heat(engine)
        runs.append(engine.state.positions.clone())
    torch.testing.assert_close(runs[0], runs[1], rtol=1e-12, atol=1e-12)


@pytest.mark.filterwarnings("ignore::fastmace.engine.graph.RecaptureBudgetWarning")
@pytest.mark.parametrize("free, releases", [(10**12, False), (1, True)])
def test_fit_releases_only_when_the_next_variant_does_not_fit(monkeypatch, free, releases):
    engine = growing_engine(monkeypatch, "fit")
    monkeypatch.setattr(engine, "_free_device_bytes", lambda: free)
    monkeypatch.setattr(engine, "_pool_bytes", lambda: 1000)  # the graph pool after each force capture
    held, _ = heat(engine, 12)
    assert (engine.info()["force_graph_releases"] > 0) is releases
    assert (max(held) == 1) is releases


def test_fit_without_a_measurement_keeps_the_variants(monkeypatch):
    """The CPU replay engine measures nothing: "fit" acts as "keep" there."""
    engine = growing_engine(monkeypatch, "fit")
    heat(engine, 12)
    assert engine.info()["force_graph_releases"] == 0


def test_the_estimate_scales_the_largest_bytes_per_edge_with_a_margin():
    assert graph.next_force_graph_fits(bytes_per_edge=100.0, edges=1000, free=10**6)
    assert not graph.next_force_graph_fits(bytes_per_edge=100.0, edges=1000, free=100_000)
    need = 100.0 * 1000 * graph.FORCE_GRAPH_MARGIN
    assert graph.next_force_graph_fits(bytes_per_edge=100.0, edges=1000, free=int(np.ceil(need)))
    assert not graph.next_force_graph_fits(bytes_per_edge=100.0, edges=1000, free=int(need) - 1)


def test_force_graphs_is_a_validated_choice():
    assert EngineConfig().force_graphs == "auto"
    for value in ("keep", "fit", "one"):
        assert dataclasses.replace(EngineConfig(), force_graphs=value).force_graphs == value
    with pytest.raises(ValueError, match="force_graphs"):
        EngineConfig(force_graphs="two")


def test_the_planner_resolves_auto():
    pytest.importorskip("mace")
    from test_planner import plan_for

    plan = plan_for("mh1-r2scan")
    assert plan.engine.force_graphs in ("fit", "keep")
    key, choice, _ = plan.decision("EngineConfig.force_graphs")
    assert choice == plan.engine.force_graphs
    explicit = plan_for("mh1-r2scan", engine=dataclasses.replace(EngineConfig(), force_graphs="one", backend="auto"))
    assert explicit.engine.force_graphs == "one"
