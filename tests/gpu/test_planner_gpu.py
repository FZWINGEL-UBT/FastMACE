"""The fixed-plan planner on the GPU (plan task 1a-T15).

- The trait facts that ``tests/cpu/test_planner.py`` builds its 19 stand-ins from equal those of the
  released checkpoints in ``FASTMACE_MODELS`` (read on the CPU with ``torch.load``).
- MH-1 (``matpes_r2scan``) on NaCl 3x3x3, NVT 300 K, 500 steps, in fresh processes: the engine that
  ``FixedPlanner`` plans and builds from main.py's defaults (``--backend auto``: hybrid FP32, in
  ``FASTMACE_HYBRID_PYTHON``; ``--backend cueq``: plain cuEq float64) against an engine assembled by
  hand with 1a-T13's helper from the same records and options. The trajectories agree within the
  run-to-run scatter (float64: 1e-12 Å, cuEq's run-time kernel choice; hybrid: three times the
  scatter of two hand-built runs), and the step times are reported.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_planner_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
CPU_PLANNER = _module("_planner_gpu_cpu_tests", REPOSITORY_ROOT / "tests/cpu/test_planner.py")

needs_models = pytest.mark.skipif(not MODELS, reason="FASTMACE_MODELS is not set")
needs_mh1 = pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")


@needs_models
def test_label_facts_are_those_of_the_released_checkpoints():
    import torch

    pytest.importorskip("graph_longrange")  # POLAR-1 checkpoints unpickle PolarMACE
    from fastmace.model import capabilities, labels

    torch.serialization.add_safe_globals([slice])
    missing, checked = [], 0
    for label, (model_class, interactions, apply_cutoff, joint) in CPU_PLANNER.LABEL_FACTS.items():
        path = Path(MODELS) / labels.LABELS[label].checkpoint.file_name
        if not path.is_file():
            missing.append(label)
            continue
        model = torch.load(path, map_location="cpu", weights_only=False)
        traits = capabilities.model_traits(model)
        assert (traits.model_class, traits.interaction_classes, traits.apply_cutoff, traits.joint_embeddings) == (
            model_class, interactions, apply_cutoff, joint), label
        assert type(model.node_embedding).__name__ == "LinearNodeEmbeddingBlock", label
        stand_in = capabilities.model_traits(CPU_PLANNER.stand_in(label))
        assert dict(traits.rewrite_support) == dict(stand_in.rewrite_support), label
        assert traits.padding_is_exact and stand_in.padding_is_exact, label
        assert traits.first_inputs_species_only == stand_in.first_inputs_species_only, label
        checked += 1
        del model
    print(f"\nchecked {checked} labels; missing checkpoints: {missing}")
    assert checked >= 1


PLANNER_CHILD = """
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.planner import fixed
from fastmace.model import compile as fc

planner = fixed.FixedPlanner(engine=EngineConfig(backend=SPEC["planner_backend"]))
hardware = planner.hardware()  # sets the allocator before the first CUDA allocation
loaded = planner.load("mh1-r2scan")
import torch
atoms = structure()
plan = planner.plan(loaded, atoms, IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=SPEC["seed"]), hardware)
engine = planner.build(plan, loaded, atoms, temperature_K=SPEC["temperature"], compile_cache=fc.CompiledModelCache())
SPEC["steps"] = 500
out = protocol(torch, engine.step, lambda: engine.state.positions, lambda: engine._graphs, engine.flush)
out["info"] = {k: v for k, v in engine.info().items() if k != "capture_memory"}
out["mode"] = engine.mode
out["records"] = list(engine.force_field.resolution.enabled)
out["options"] = list(fixed.enabled_options(plan))
out["backend"], out["dtype"] = plan.engine.backend, plan.precision.dtype
out["allocator"] = engine.process["allocator"]
out["plan"] = plan.describe()
out["hardware"] = [hardware.name, list(hardware.compute_capability), hardware.memory_bytes]
out["torch"] = torch.__version__
print(json.dumps(out))
"""


def planner_run(backend, python):
    spec = ENGINE.spec("hybrid" if backend == "auto" else backend)
    spec["planner_backend"] = backend
    return ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + PLANNER_CHILD, python=python)


def direct_run(planned, python):
    """1a-T13's hand-built engine with the records, receiver order and step mode of the planned one."""
    spec = ENGINE.spec(planned["backend"], modes=[("pipelined", 500)])
    spec["rewrites"] = planned["records"]
    spec["sort"] = "on" if "sort_receivers" in planned["options"] else "off"
    assert spec["dtype"] == planned["dtype"]
    return ENGINE.fastmace_run(spec, python=python)["modes"]["pipelined"]


@needs_mh1
def test_mh1_plain_cueq_float64_planned_equals_hand_built():
    planned = planner_run("cueq", sys.executable)
    direct = direct_run(planned, sys.executable)
    # main.py --backend cueq: no MH-1 or linear rewrites (factor_species and block_linear follow the hybrid);
    # the strain patch is process-wide.
    assert planned["records"] == ["single_graph_strain"]
    assert planned["options"] == ["single_graph_strain", "gpu_neighbour_list"]
    assert planned["mode"] == "pipelined" and planned["allocator"] == "expandable_segments:True"
    summary = {"vs_hand_built_500": ENGINE.deviation(planned["positions"], direct["positions"]),
               "ms_per_step": {"planned": planned["ms_per_step"], "hand_built": direct["ms_per_step"]},
               "edge_sizes": [planned["info"]["edge_sizes_used"], direct["info"]["edge_sizes_used"]],
               "rebuilds": [planned["info"]["neighbour_rebuilds"], direct["info"]["neighbour_rebuilds"]],
               "records": planned["records"], "hardware": planned["hardware"], "torch": planned["torch"],
               "plan": planned["plan"]}
    ENGINE.report("1a-T15-mh1-cueq-float64-planned", summary)
    assert summary["vs_hand_built_500"] <= ENGINE.FLOAT64_LIMIT, summary["vs_hand_built_500"]
    assert planned["info"]["edge_sizes_used"] == direct["info"]["edge_sizes_used"]
    assert planned["info"]["neighbour_rebuilds"] == direct["info"]["neighbour_rebuilds"]
    assert planned["ms_per_step"] < 1.25 * direct["ms_per_step"], summary["ms_per_step"]


@ENGINE.needs_oeq
@needs_mh1
def test_mh1_hybrid_float32_planned_equals_hand_built():
    planned = planner_run("auto", HYBRID_PYTHON)
    first = direct_run(planned, HYBRID_PYTHON)
    second = direct_run(planned, HYBRID_PYTHON)
    # The production set: every MH-1 rewrite, the linear family, streaming on receiver-sorted edges.
    assert set(planned["records"]) == set(ENGINE.HYBRID_REWRITES)
    assert planned["mode"] == "pipelined" and (planned["backend"], planned["dtype"]) == ("hybrid", "float32")
    scatter = ENGINE.deviation(second["positions"], first["positions"])
    summary = {"vs_hand_built_500": ENGINE.deviation(planned["positions"], first["positions"]),
               "hand_built_scatter_500": scatter,
               "ms_per_step": {"planned": planned["ms_per_step"], "hand_built": [first["ms_per_step"],
                                                                                  second["ms_per_step"]]},
               "edge_sizes": [planned["info"]["edge_sizes_used"], first["info"]["edge_sizes_used"]],
               "rebuilds": [planned["info"]["neighbour_rebuilds"], first["info"]["neighbour_rebuilds"]],
               "records": planned["records"], "hardware": planned["hardware"], "torch": planned["torch"],
               "plan": planned["plan"]}
    ENGINE.report("1a-T15-mh1-hybrid-float32-planned", summary)
    limit = max(3 * scatter, 1e-6)
    assert summary["vs_hand_built_500"] <= limit, (summary["vs_hand_built_500"], limit)
    assert planned["ms_per_step"] < 1.25 * min(first["ms_per_step"], second["ms_per_step"]), summary["ms_per_step"]

