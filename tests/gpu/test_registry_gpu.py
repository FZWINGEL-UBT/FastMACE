"""The runtime-state checks of SPEC §12.2 on the GPU (plan task 1b-T04), with released models.

Each case runs in a fresh process: the fixed planner plans and builds the production engine for
NaCl 3x3x3 at 300 K, as ``main.py`` does, and the child then checks:

- **capture warm-ups leave no trace**: with every sticky flag set, each stage the run uses (AC, C,
  ACP, every B bucket, the speculative ACB where the mode has it) is captured again, and the full
  registered state is identical before and after, at the same addresses;
- **the registry covers what the stages mutate**: the same stages, run eagerly (captured replays do
  not move version counters), mutate no tensor reachable from the engine that the registry lacks.

Cases: MH-1 hybrid pipelined NVT (``FASTMACE_HYBRID_PYTHON``), MH-1 plain cuEq float64, MH-1 NPT
BAOAB with a full cell (hybrid), and POLAR-1-S (hybrid).
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_registry_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.engine.registry import unregistered_mutations
from fastmace.planner import fixed
from fastmace.model import compile as fc

planner = fixed.FixedPlanner(engine=EngineConfig(backend=SPEC["planner_backend"]))
hardware = planner.hardware()
loaded = planner.load(SPEC["label"])
import torch
atoms = structure()
options = dict(timestep_fs=0.5, friction=0.02, seed=SPEC["seed"])
if SPEC["ensemble"] == "npt":
    options.update(ensemble="npt", pressure_GPa=0.5, cell_mode="full")
plan = planner.plan(loaded, atoms, IntegratorConfig(**options), hardware)
engine = planner.build(plan, loaded, atoms, temperature_K=SPEC["temperature"], compile_cache=fc.CompiledModelCache())
engine.run(20)
engine.flush()
torch.cuda.synchronize()

def keys():
    out = [key for key in engine._graphs]
    for key in ("AC", "C"):
        if key not in out:
            out.append(key)
    return out

# Capture warm-ups leave no trace, sticky flags included.
engine.integrator.prepare_step(engine.state)
for name, (policy, tensor) in engine.registry.entries().items():
    if policy.sticky:
        tensor.fill_(1)
bad, captured = [], []
for key in keys():
    before = {name: (tensor.data_ptr(), tensor.clone()) for name, (policy, tensor) in engine.registry.entries().items()
              if policy.capture}
    engine._graphs.pop(key, None)
    engine._capture(key)
    torch.cuda.synchronize()
    after = engine.registry.tensors()
    captured.append(str(key))
    bad += [f"{key}:{name}" for name, (pointer, value) in before.items()
            if name not in after or after[name].data_ptr() != pointer or not torch.equal(after[name], value)]
sticky = sorted(name for name, (policy, _) in engine.registry.entries().items() if policy.sticky)

# The registry covers every tensor the stages mutate (eager stages, fresh flags).
for name, (policy, tensor) in engine.registry.entries().items():
    if policy.sticky:
        tensor.zero_()
engine._clear_graphs()
engine.capture = False

def exercise():
    engine.run(3)
    engine.flush()
    engine._rebuild()
    engine._run("C")
    for key in keys():
        engine.integrator.prepare_step(engine.state)
        engine._stage(key)()
    torch.cuda.synchronize()

unregistered = unregistered_mutations(engine, exercise)
print(json.dumps({"warmup_mismatches": bad, "captured": captured, "sticky": sticky, "unregistered": unregistered,
                  "mode": engine.mode, "backend": plan.engine.backend, "dtype": plan.precision.dtype,
                  "registered": sorted(engine.registry.tensors()), "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name()}))
"""


def registry_run(label, backend, python, ensemble="nvt"):
    spec = ENGINE.spec("hybrid" if backend == "auto" else backend)
    spec.update(planner_backend=backend, label=label, ensemble=ensemble)
    return ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD, python=python)


def check(result, name):
    ENGINE.report(f"1b-T04-registry-{name}", result)
    assert result["warmup_mismatches"] == [], result["warmup_mismatches"]
    assert result["unregistered"] == [], result["unregistered"]
    assert {"engine.flags", "engine.pipe_flags"} <= set(result["sticky"])


@ENGINE.needs_oeq
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_mh1_hybrid_pipelined_nvt():
    result = registry_run("mh1-r2scan", "auto", ENGINE.HYBRID_PYTHON)
    assert (result["backend"], result["mode"]) == ("hybrid", "pipelined")
    check(result, "mh1-hybrid-nvt")


@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_mh1_plain_cueq_float64():
    result = registry_run("mh1-r2scan", "cueq", sys.executable)
    assert (result["backend"], result["dtype"]) == ("cueq", "float64")
    check(result, "mh1-cueq-float64")


@ENGINE.needs_oeq
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_mh1_npt_baoab_full_cell():
    result = registry_run("mh1-r2scan", "auto", ENGINE.HYBRID_PYTHON, ensemble="npt")
    assert {"integrator.p_eps", "force_field.neighbours.build_cell"} & set(result["registered"])
    check(result, "mh1-hybrid-npt-full")


@ENGINE.needs_oeq
@pytest.mark.skipif(not ENGINE.MODELS, reason="FASTMACE_MODELS is not set")
def test_polar_1_s():
    result = registry_run("polar-1-s", "auto", ENGINE.HYBRID_PYTHON)
    assert "force_field.polar.overflow" in result["sticky"]
    check(result, "polar-1-s")
