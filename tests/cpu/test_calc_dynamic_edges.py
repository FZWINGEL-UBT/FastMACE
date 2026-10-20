"""Evaluation plans compile once when the edge count changes (plan task 2-T13; SPEC §8.2).

1b-T18 measured a second compile at ``fm.calc``'s second call (4.5 s plain cuEq, 19 s hybrid): the
edge count changed and dynamo's automatic dynamic shapes compiled the model again with a symbolic
edge dimension. Evaluation plans now set ``EngineConfig.dynamic_edges``, and their force field
marks the edge dimension dynamic from the first call:

- the evaluation plan sets ``dynamic_edges`` when it compiles; MD (graph tier) plans do not;
- ``mark_edges_dynamic`` makes one compiled function serve every edge count;
- the force field marks its edges and keys its compiled model by ``dynamic_edges``;
- slow: a compiled tiny MACE compiles once across calls with three edge counts and matches the
  uncompiled model; without ``dynamic_edges`` the same calls compile again.
"""

import dataclasses
from unittest import mock

import numpy as np
import pytest
import test_planner as tp
import torch

from _support.structures import nacl
from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.model import compile as fc
from fastmace.planner.minimal import MinimalPlanner, SoftwareTraits
from fastmace.planner.workload import Workload

pytest.importorskip("mace")

GPU = tp.GPU
NEIGHBOURS = NeighbourConfig(skin=0.5, sort_receivers="off", gpu_neighbour_list="off")
Z_VALUES = [8, 11, 17]


def minimal(**configs):
    software = SoftwareTraits(oeq=(True, "openequivariance is installed"), triton=(True, "triton is importable"),
                              nvrtc=(True, "NVRTC loads"), fused_cueq=(True, "cuequivariance_ops_torch imports"),
                              mace_imported_first=())
    return MinimalPlanner(software=software, process=constraints.ProcessRecord(), environ={},
                          torch_version=tp.TORCH_2_14, **configs)


def calculator():
    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    return tiny_mace_calculator(Z_VALUES, **FAMILIES["mh1"])


def strained(atoms, factor):
    """``atoms`` with its cell and positions scaled by ``factor`` (another edge count)."""
    out = atoms.copy()
    out.set_cell(atoms.cell.array * factor, scale_atoms=True)
    return out


# --------------------------------------------------------------------------- the plan


def test_evaluation_plans_compile_with_dynamic_edges():
    plan = minimal().plan_evaluation_traits(tp.traits("mh1-r2scan"), nacl(), GPU, precision="float32")
    assert plan.engine.compile_mode == "default" and plan.engine.dynamic_edges is True
    assert plan.decision("EngineConfig.dynamic_edges")[1] == "on"


def test_uncompiled_evaluation_plans_leave_dynamic_edges_off():
    plan = minimal(engine=EngineConfig(device="cpu", compile_mode="none")).plan_evaluation_traits(
        tp.traits("mh1-r2scan"), nacl(), tp.CPU, precision="float32")
    assert plan.engine.dynamic_edges is False
    assert plan.decision("EngineConfig.dynamic_edges")[1] == "off"


def test_md_plans_keep_static_edges():
    """The graph tier pads edges to buckets and captures graphs: its plans are unchanged."""
    plan = minimal().plan_traits(tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(),
                                 Workload(max_temperature_K=300.0), GPU, precision="float32")
    assert plan.engine.dynamic_edges is False


# --------------------------------------------------------------------------- marking


def test_marked_edges_compile_once_for_every_edge_count():
    from torch._dynamo.testing import CompileCounter

    def edge_sum(data):
        sender, receiver = data["edge_index"]
        return (data["positions"][receiver] - data["positions"][sender] + data["shifts"]).sum()

    def data(edges):
        generator = torch.Generator().manual_seed(edges)
        return {"positions": torch.randn(8, 3, generator=generator),
                "edge_index": torch.randint(0, 8, (2, edges), generator=generator),
                "shifts": torch.randn(edges, 3, generator=generator),
                "unit_shifts": torch.zeros(edges, 3)}

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(edge_sum, backend=counter)
    for edges in (40, 47, 53):
        inputs = data(edges)
        fc.mark_edges_dynamic(inputs)
        torch.testing.assert_close(compiled(inputs), edge_sum(inputs))
    assert counter.frame_count == 1
    torch._dynamo.reset()


# --------------------------------------------------------------------------- the force field


def test_force_field_marks_edges_and_keys_its_compiled_model_by_dynamic_edges():
    from fastmace.model.forces import MACEForceField

    calc, atoms = calculator(), nacl(repeat=2, rattle=0.1, seed=8)
    atoms.numbers[3] = 8
    compiled_for = []

    def fake_compile(model, mode, **kwargs):
        compiled_for.append(kwargs["engine"].dynamic_edges)
        private = fc.private_copy(model)
        if kwargs.get("prepare") is not None:
            kwargs["prepare"](private)
        return private

    cache, static = fc.CompiledModelCache(), EngineConfig(compile_mode="default")
    dynamic = dataclasses.replace(static, dynamic_edges=True)
    with mock.patch.object(fc, "compile_model", side_effect=fake_compile), \
            mock.patch.object(fc, "mark_edges_dynamic", wraps=fc.mark_edges_dynamic) as marked:
        plain = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=static, compile_cache=cache)
        assert marked.call_count == 0
        field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=dynamic, compile_cache=cache,
                               compute_stress=True)
        assert marked.call_count == 1
        field.evaluate(strained(atoms, 1.02), stress=True)
        assert marked.call_count == 2 and marked.call_args.args[0] is field.data
        uncompiled = MACEForceField(calc, atoms, neighbours=NEIGHBOURS,
                                    engine=EngineConfig(compile_mode="none", dynamic_edges=True))
        uncompiled.evaluate(atoms)
        assert marked.call_count == 2  # nothing to mark without a compiled model
    assert compiled_for == [False, True] and len(cache) == 2 and plain.model is not field.model


@pytest.mark.slow
@pytest.mark.parametrize("dynamic_edges", [True, False])
def test_compiled_evaluation_compiles_once_across_edge_counts(dynamic_edges):
    """torch.compile on the CPU: three calls with three edge counts (strained cells)."""
    pytest.importorskip("torch._inductor")
    from torch._dynamo.utils import counters

    from fastmace.model.forces import MACEForceField

    calc, atoms = calculator(), nacl(repeat=2, rattle=0.1, seed=8)
    atoms.numbers[3] = 8
    plain = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=True)
    torch._dynamo.reset()
    counters.clear()
    compiled = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=True,
                              engine=EngineConfig(compile_mode="default", dynamic_edges=dynamic_edges))
    graphs, edge_counts = [], []
    for factor in (1.0, 0.97, 1.03):
        current = strained(atoms, factor)
        got = compiled.evaluate(current, stress=True)
        edge_counts.append(int(compiled.neighbours.count))
        graphs.append(counters["stats"]["unique_graphs"])
        expected = plain.evaluate(current, stress=True)
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-10, atol=0)
        for a, b in zip(got[1:], expected[1:]):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
    assert len(set(edge_counts)) == 3, edge_counts
    if dynamic_edges:
        assert graphs[0] > 0 and graphs[1:] == [graphs[0]] * 2, graphs  # no compile after the first call
    else:
        assert graphs[1] > graphs[0], graphs  # 1b-T18's second compile
    torch._dynamo.reset()
