"""Tiny-model versions of two MH-1 engine tests, for the fast tier (plan task 1b-T25).

The MH-1 tests are marked ``slow`` (1b-T24); these keep their behaviours in ``-m "not slow"``.
A tiny MH-1-like model (``_support.tiny_models``, r_max 4 Angstrom) on NaCl 2x2x2, in fresh processes:

- no compile at a later edge bucket (``test_engine_gpu.py::test_no_compile_at_a_later_edge_bucket_plain_cueq``):
  after warm-up, a run that reaches a new padded edge size captures it without a dynamo compile, from a
  fresh Inductor cache, with a foreign capture-safe ``mask_head`` installed as the research engine does.
  Plain cuEq only: a tiny hybrid version costs another 50 s, and the MH-1 hybrid test stays in the full tier;
- a planned engine equals a hand-built one
  (``test_planner_gpu.py::test_mh1_plain_cueq_float64_planned_equals_hand_built_and_minimal``): ``FixedPlanner``
  with main.py's ``--backend cueq`` defaults against an engine assembled by hand with 1a-T13's
  settings, float64, within 1e-12 Angstrom, with the same edge sizes and rebuilds.

Each fresh process costs about 25 s with warm caches (loading, conversion and the dynamo trace); the
edge-bucket test needs another 25 s, because it compiles from an empty Inductor cache.
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch

from _support.tiny_models import FAMILIES, tiny_mace_model


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_tiny_engine_gpu_helpers", Path(__file__).parent / "test_engine_gpu.py")


def tiny_mh1(directory):
    path = directory / "tiny-mh1.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    return str(path)


# NaCl 2x2x2 (64 atoms). The tiny model's cutoff (4 Angstrom) lies just above the second shell
# (3.99 Angstrom), so the edge count moves by whole buckets as atoms vibrate.
STRUCTURE = """
def structure(temperature):
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(11))
    return atoms
"""


# --------------------------------------------------------------------------- no compile at a later edge bucket

BUCKET_CHILD = """
import json, time
import numpy as np
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.planner import fixed

planner = fixed.FixedPlanner(engine=EngineConfig(backend=SPEC["backend"]))
hardware = planner.hardware()
loaded = planner.load(SPEC["path"], trusted=True)
import torch
from torch._dynamo.utils import counters

atoms = structure(900.0)
plan = planner.plan(loaded, atoms, IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=20261018), hardware)
engine = planner.build(plan, loaded, atoms, temperature_K=900.0)

def compiled():
    return {"unique_graphs": counters["stats"]["unique_graphs"], "frames_ok": counters["frames"]["ok"]}

for _ in range(SPEC["warmup"]):
    engine.step()
engine.flush()
# What the research engine's use_capture_safe_mask_head does before each of its captures in a
# shared process: another capture-safe mask_head in mace.modules.blocks.
from mace.modules import blocks
from fastmace.rewrites import engine as engine_rewrites

def foreign_mask_head(x, heads, num_heads):
    return engine_rewrites.capture_safe_mask_head(x, heads, num_heads)
blocks.mask_head = foreign_mask_head
after_warmup, sizes_after_warmup = compiled(), sorted(engine.edge_sizes_used)
events, seen = [], set(engine.edge_sizes_used)
start = time.perf_counter()
for n in range(SPEC["warmup"] + 1, SPEC["steps"] + 1):
    engine.step()
    engine.flush()
    if set(engine.edge_sizes_used) != seen:
        events.append({"step": n, "new_sizes": sorted(set(engine.edge_sizes_used) - seen), "after": compiled()})
        seen = set(engine.edge_sizes_used)
torch.cuda.synchronize()
print(json.dumps({"after_warmup": after_warmup, "end": compiled(), "sizes_after_warmup": sizes_after_warmup,
                  "sizes": sorted(engine.edge_sizes_used), "events": events, "edge_bucket": engine.edge_bucket,
                  "initial_edges": engine.initial_edges, "run_s": time.perf_counter() - start}))
"""


def _bucket_crossing(backend, python, directory):
    spec = {"backend": backend, "path": tiny_mh1(directory), "warmup": 20, "steps": 200}
    # A fresh Inductor cache: dynamo's automatic-dynamic profile persists there, and a warm one would
    # hide a compile that a cold process does. The run's Triton cache (kernel binaries keyed by their
    # source) and FastMACE cache change no compile decision and stay shared.
    out = ENGINE.run_fresh(f"import numpy as np\nSPEC = {spec!r}\n" + STRUCTURE + BUCKET_CHILD, python=python,
                           extra_env={"TORCHINDUCTOR_CACHE_DIR": str(directory / "inductor")})
    print(json.dumps(out, indent=1))
    # A new padded size needs a capture, but no compile: warm-up saw the exact initial count and a
    # padded size, so the edge dimension is dynamic; the capture puts FastMACE's mask_head back first.
    assert out["events"], "the run never reached a new edge bucket after warm-up"
    assert out["end"] == out["after_warmup"], out


def test_no_compile_at_a_later_edge_bucket_tiny_plain_cueq(tmp_path):
    _bucket_crossing("cueq", sys.executable, tmp_path)


# --------------------------------------------------------------------------- planned equals hand-built

PLANNER_CHILD = """
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.planner import fixed
from fastmace.model import compile as fc

planner = fixed.FixedPlanner(engine=EngineConfig(backend="cueq"))
hardware = planner.hardware()  # sets the allocator before the first CUDA allocation
loaded = planner.load(SPEC["path"], trusted=True)
import torch
from fastmace.engine.graph import GraphEngine
from fastmace.integrators.langevin import Langevin
from fastmace.model.forces import MACEForceField
from fastmace.rewrites import registry as reg
from fastmace.state import SimulationState

# 1a-T13's hand-built engine runs ASE's projection (3-T22c pins it in the plan too)
integrator = IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=SPEC["seed"], fixcm_projection="ase")
temperature = SPEC["temperature"]
cache = fc.CompiledModelCache()
plan = planner.plan(loaded, structure(temperature), integrator, hardware)
planned = planner.build(plan, loaded, structure(temperature), temperature_K=temperature, compile_cache=cache)
records = list(planned.force_field.resolution.enabled)

# By hand, as 1a-T13's helper builds it: the planned rewrite records, main.py's neighbour and step settings.
atoms = structure(temperature)
rewrites = {name: "on" if name in records else "off" for name in reg.default_registry().names()}
field = MACEForceField(loaded, atoms, rewrites=rewrites, engine=EngineConfig(compile_mode="default"),
                       neighbours=NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="on"),
                       compile_cache=cache)
state = SimulationState.from_atoms(atoms, "cuda:0", SPEC["seed"])
dyn = Langevin(state, integrator, temperature_K=temperature)
direct = GraphEngine(EngineConfig(compile_mode="default", pipelined="on"), field, dyn, state, pad_edges=True)

out = {"records": records, "options": list(fixed.enabled_options(plan)), "modes": [planned.mode, direct.mode],
       "allocator": planned.process["allocator"], "backend": plan.engine.backend, "dtype": plan.precision.dtype}
for name, engine in (("planned", planned), ("direct", direct)):
    out[name] = protocol(torch, engine.step, lambda: engine.state.positions, lambda: engine._graphs, engine.flush)
    out[name]["info"] = {k: v for k, v in engine.info().items() if k != "capture_memory"}
print(json.dumps(out))
"""


def test_tiny_plain_cueq_float64_planned_equals_hand_built(tmp_path):
    spec = ENGINE.spec("cueq")
    spec.update(path=tiny_mh1(tmp_path), steps=200, timed=100)
    # COMMON's protocol; STRUCTURE replaces its 3x3x3 structure().
    common = ENGINE.COMMON.format(spec=spec)
    out = ENGINE.run_fresh(common + STRUCTURE + PLANNER_CHILD)
    planned, direct = out["planned"], out["direct"]
    summary = {"vs_hand_built_200": ENGINE.deviation(planned["positions"], direct["positions"]),
               "ms_per_step": [planned["ms_per_step"], direct["ms_per_step"]],
               "edge_sizes": [planned["info"]["edge_sizes_used"], direct["info"]["edge_sizes_used"]],
               "rebuilds": [planned["info"]["neighbour_rebuilds"], direct["info"]["neighbour_rebuilds"]]}
    print(json.dumps(summary, indent=1))
    # main.py --backend cueq: no MH-1 or linear rewrites; the strain patch is process-wide.
    assert out["records"] == ["single_graph_strain"]
    assert out["options"] == ["single_graph_strain", "gpu_neighbour_list"]
    assert (out["backend"], out["dtype"]) == ("cueq", "float64")
    assert out["modes"] == ["pipelined", "pipelined"] and out["allocator"] == "expandable_segments:True"
    assert summary["vs_hand_built_200"] <= ENGINE.FLOAT64_LIMIT, summary
    assert planned["info"]["edge_sizes_used"] == direct["info"]["edge_sizes_used"]
    assert planned["info"]["neighbour_rebuilds"] == direct["info"]["neighbour_rebuilds"]
