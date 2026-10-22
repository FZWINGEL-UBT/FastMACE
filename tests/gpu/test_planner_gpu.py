"""The fixed-plan and minimal planners on the GPU (plan tasks 1a-T15, 1b-T23; merged in 1b-T27).

- The trait facts that ``tests/cpu/test_planner.py`` builds its 19 stand-ins from equal those of the
  released checkpoints in ``FASTMACE_MODELS`` (read on the CPU with ``torch.load``).
- MH-1 (``matpes_r2scan``) on NaCl 3x3x3, NVT 300 K, 500 steps, one fresh process per backend
  (``--backend auto``: hybrid FP32, in ``FASTMACE_HYBRID_PYTHON``; ``--backend cueq``: plain cuEq
  float64), with three engines from the same seed:
  - the engine ``FixedPlanner`` plans and builds from main.py's defaults;
  - an engine assembled by hand with 1a-T13's settings from the planned records and options;
  - the engine ``MinimalPlanner`` plans and builds from an ``fm.ForceField`` (the public path of
    ``Simulation`` and ``fm.md``).
  The hand-built and the minimal engine each follow the planned one within the run-to-run scatter
  (float64: 1e-12 Å, cuEq's run-time kernel choice; hybrid: three times the scatter between the
  planned engine and a second planned engine). They have the same rewrite records, options, edge
  sizes and rebuilds (full tier). The planned engine is less than 25 % slower than the hand-built
  one (acceptance tier, which runs on an idle GPU; 2-T14d): on a shared GPU the step times are not
  comparable. Both use the same run when they run in one process.
"""

import functools
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


THREE_ENGINES_CHILD = """
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.model import loading
from fastmace.model import compile as fc
from fastmace.model.forcefield import ForceField
from fastmace.planner import fixed, minimal
from fastmace.planner.workload import Workload

# fm.load first: it prepares the OEq runtime before MACE is imported, which the hybrid needs. The fixed
# planner then sets the allocator before the first CUDA allocation.
pristine = loading.load_pristine("mh1-r2scan")
planner = fixed.FixedPlanner(engine=EngineConfig(backend=SPEC["planner_backend"]))
hardware = planner.hardware()
loaded = planner.load("mh1-r2scan")
import torch
from fastmace.engine.graph import GraphEngine
from fastmace.integrators.langevin import Langevin
from fastmace.model.forces import MACEForceField
from fastmace.rewrites import registry as reg
from fastmace.state import SimulationState

SPEC["steps"] = 500
# The hand-built 1a-T13 engine runs ASE's projection; the plans pin it (3-T22c).
integrator = IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=SPEC["seed"], fixcm_projection="ase")
plan = planner.plan(loaded, structure(), integrator, hardware)
fixed_cache = fc.CompiledModelCache()


def measure(engine, **extra):
    out = protocol(torch, engine.step, lambda: engine.state.positions, lambda: engine._graphs, engine.flush)
    out["info"] = {k: v for k, v in engine.info().items() if k != "capture_memory"}
    out["mode"] = engine.mode
    out["records"] = list(engine.force_field.resolution.enabled)
    out.update(extra)
    return out


def planned_run():
    engine = planner.build(plan, loaded, structure(), temperature_K=SPEC["temperature"], compile_cache=fixed_cache)
    out = measure(engine, options=list(fixed.enabled_options(plan)), backend=plan.engine.backend,
                  dtype=plan.precision.dtype, allocator=engine.process["allocator"])
    del engine
    gc.collect(); torch.cuda.empty_cache()
    return out


result = {"planned": planned_run(), "plan": plan.describe(), "torch": torch.__version__,
          "hardware": [hardware.name, list(hardware.compute_capability), hardware.memory_bytes]}

# By hand (1a-T13's helper): the planned records, receiver order and step mode; main.py's settings.
records, options = result["planned"]["records"], result["planned"]["options"]
atoms = structure()
rewrites = {name: "on" if name in records else "off" for name in reg.default_registry().names()}
neighbours = NeighbourConfig(skin=0.3, sort_receivers="on" if "sort_receivers" in options else "off",
                             gpu_neighbour_list="on")
field = MACEForceField(loaded, atoms, rewrites=rewrites, engine=EngineConfig(compile_mode="default"),
                       neighbours=neighbours, compile_cache=fc.CompiledModelCache())
state = SimulationState.from_atoms(atoms, "cuda:0", SPEC["seed"])
dyn = Langevin(state, IntegratorConfig(timestep_fs=0.5, friction=0.02, fixcm_projection="ase"),
               temperature_K=SPEC["temperature"])
engine = GraphEngine(EngineConfig(compile_mode="default", pipelined="on"), field, dyn, state, pad_edges=True)
result["hand_built"] = measure(engine)
del engine, field, state, dyn
gc.collect(); torch.cuda.empty_cache()

if SPEC["repeat"]:
    result["planned_repeat"] = planned_run()

# MinimalPlanner from an fm.ForceField, as Simulation and fm.md build it.
minimal_planner = minimal.MinimalPlanner(engine=EngineConfig(backend=SPEC["planner_backend"]))
minimal_hardware = minimal_planner.hardware()
atoms = structure()
force_field = ForceField(pristine, atoms, precision=SPEC["precision"])
minimal_integrator = IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=SPEC["seed"],
                                      temperature_K=SPEC["temperature"], fixcm_projection="ase")
minimal_plan = minimal_planner.plan(force_field, atoms, minimal_integrator,
                                    Workload(max_temperature_K=SPEC["temperature"]), minimal_hardware)
engine = minimal_planner.build(minimal_plan, force_field, atoms, temperature_K=SPEC["temperature"])
result["minimal"] = measure(engine, options=list(fixed.enabled_options(minimal_plan)),
                            backend=minimal_plan.engine.backend, dtype=minimal_plan.precision.dtype)
print(json.dumps(result))
"""

ENGINES = ("planned", "hand_built", "minimal")


def three_engines(backend, python):
    """The planned, hand-built and minimal engines (for the hybrid, a second planned one) in one process."""
    hybrid = backend == "auto"
    spec = ENGINE.spec("hybrid" if hybrid else backend)
    spec.update(planner_backend=backend, precision="float32" if hybrid else "float64", repeat=hybrid)
    result = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + THREE_ENGINES_CHILD, python=python)
    assert spec["dtype"] == result["planned"]["dtype"]
    return result


@functools.cache
def three_engines_once(backend, python):
    """``three_engines``, once per process: the step-time test reuses the numerical test's run."""
    return three_engines(backend, python)


def summary(result, **extra):
    planned = result["planned"]
    return {"hand_built_vs_planned_500": ENGINE.deviation(planned["positions"], result["hand_built"]["positions"]),
            "minimal_vs_planned_500": ENGINE.deviation(result["minimal"]["positions"], planned["positions"]),
            "ms_per_step": {name: result[name]["ms_per_step"] for name in ENGINES},
            "edge_sizes": {name: result[name]["info"]["edge_sizes_used"] for name in ENGINES},
            "rebuilds": {name: result[name]["info"]["neighbour_rebuilds"] for name in ENGINES},
            "records": planned["records"], "options": planned["options"], "hardware": result["hardware"],
            "torch": result["torch"], "plan": result["plan"], **extra}


def same_plan(result):
    """The hand-built and the minimal engine run what the planned one runs."""
    planned, direct, minimal = (result[name] for name in ENGINES)
    assert planned["mode"] == direct["mode"] == "pipelined"
    for other in (direct, minimal):
        assert other["info"]["edge_sizes_used"] == planned["info"]["edge_sizes_used"]
        assert other["info"]["neighbour_rebuilds"] == planned["info"]["neighbour_rebuilds"]
    assert minimal["records"] == planned["records"] and minimal["options"] == planned["options"]
    assert (minimal["backend"], minimal["dtype"], minimal["mode"]) == (planned["backend"], planned["dtype"],
                                                                      planned["mode"])


@pytest.mark.slow
@needs_mh1
def test_mh1_plain_cueq_float64_planned_equals_hand_built_and_minimal():
    result = three_engines_once("cueq", sys.executable)
    planned = result["planned"]
    report = summary(result)
    ENGINE.report("1b-T27-mh1-cueq-float64-three-engines", report)
    # main.py --backend cueq: no MH-1 or linear rewrites (factor_species and block_linear follow the hybrid);
    # the strain patch is process-wide.
    assert planned["records"] == ["single_graph_strain"]
    assert planned["options"] == ["single_graph_strain", "gpu_neighbour_list"]
    assert planned["allocator"] == "expandable_segments:True"
    same_plan(result)
    assert report["hand_built_vs_planned_500"] <= ENGINE.FLOAT64_LIMIT, report
    assert report["minimal_vs_planned_500"] <= ENGINE.FLOAT64_LIMIT, report


def assert_planned_as_fast_as_hand_built(result):
    """The planned engine is less than 25 % slower than the hand-built one (needs an idle GPU)."""
    times = {name: result[name]["ms_per_step"] for name in ENGINES}
    assert times["planned"] < 1.25 * times["hand_built"], times


@pytest.mark.slow
@pytest.mark.acceptance
@needs_mh1
def test_mh1_plain_cueq_float64_planned_is_as_fast_as_hand_built():
    assert_planned_as_fast_as_hand_built(three_engines_once("cueq", sys.executable))


@pytest.mark.slow
@ENGINE.needs_oeq
@needs_mh1
def test_mh1_hybrid_float32_planned_equals_hand_built_and_minimal():
    result = three_engines_once("auto", HYBRID_PYTHON)
    planned = result["planned"]
    scatter = ENGINE.deviation(result["planned_repeat"]["positions"], planned["positions"])
    report = summary(result, planned_scatter_500=scatter)
    ENGINE.report("1b-T27-mh1-hybrid-float32-three-engines", report)
    # The production set: every MH-1 rewrite, the linear family, streaming on receiver-sorted edges.
    assert set(planned["records"]) == set(ENGINE.HYBRID_REWRITES)
    assert (planned["backend"], planned["dtype"]) == ("hybrid", "float32")
    same_plan(result)
    limit = max(3 * scatter, 1e-6)
    assert report["hand_built_vs_planned_500"] <= limit, (report["hand_built_vs_planned_500"], limit)
    assert report["minimal_vs_planned_500"] <= limit, (report["minimal_vs_planned_500"], limit)


@pytest.mark.slow
@pytest.mark.acceptance
@ENGINE.needs_oeq
@needs_mh1
def test_mh1_hybrid_float32_planned_is_as_fast_as_hand_built():
    assert_planned_as_fast_as_hand_built(three_engines_once("auto", HYBRID_PYTHON))
