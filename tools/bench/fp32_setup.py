"""Setup time of ``fm.md`` per stage, from the call to the first MD step (fp32 profile, 28 October 2026).

Runs ``fastmace.md`` (MH-1 float32 default plan, NVT Langevin at --timestep-fs, --steps steps, default output)
on one structure with timers wrapped around the setup stages (2-T19's method, extended to the stages added
since: model templates, radial-table fit and certificate, OEq NVRTC/fused-forward kernels, dynamic edges).
Each timer records its wall time with its parent timer, so stages nest (``a > b``); a CUDA synchronisation
runs before and after each one, except while a stream is capturing a CUDA graph. Also recorded: process start
to ``fm.md`` (imports, OEq runtime), Simulation's own ``_setup_s`` record, torch.compile's phase times
(``torch._dynamo.utils.compile_times``) and counters, the first step and the steady state of the run with
output (wall ms per step over the last --timed-steps steps).

Cold against warm: run once with fresh, empty cache directories (FASTMACE_CACHE, TORCHINDUCTOR_CACHE_DIR,
TRITON_CACHE_DIR, MACE_COMPILE_CACHE) and again with the same ones; the launcher sets them. ``--cprofile``
adds a cProfile of the whole call (slower; a separate process, for locating time the timers miss).

Measurement only. Writes setup.json (and cprofile.txt) into --output.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROCESS_T0 = time.perf_counter()
import fastmace  # noqa: E402,F401  (before torch: soft file locks, SPEC §14)

IMPORT_FASTMACE_S = time.perf_counter() - PROCESS_T0
TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

# (module, attribute path, label). Host-side functions only: nothing torch.compile traces (a timer there
# would add graph breaks and change what is compiled).
TARGETS = [
    ("fastmace.quick", "md", "fm.md (total)"),
    ("fastmace.model.loading", "load_pristine", "load: load_pristine"),
    ("fastmace.model.loading", "sha256_of", "load: sha256_of"),
    ("fastmace.model.loading", "resolve_checkpoint", "load: resolve_checkpoint"),
    ("fastmace.model.loading", "load", "load: loading.load (MACE calculator, conversion)"),
    ("fastmace.model.loading", "check_oeq", "load: check_oeq"),
    ("fastmace.model.loading", "check_cueq", "load: check_cueq"),
    ("fastmace.model.loading", "prepare_hybrid", "OEq: prepare_hybrid"),
    ("fastmace.model.loading", "prepare_hybrid_if_possible", "OEq: prepare_hybrid_if_possible"),
    ("fastmace.kernels._oeq_runtime", "reuse_extension", "OEq: reuse_extension (build or cache hit)"),
    ("fastmace.kernels._oeq_runtime", "load_prebuilt_module", "OEq: load_prebuilt_module"),
    ("fastmace.simulation", "Simulation.__init__", "Simulation.__init__"),
    ("fastmace.simulation", "Simulation._build_for", "Simulation._build_for"),
    ("fastmace.simulation", "Simulation._validate", "Simulation._validate"),
    ("fastmace.simulation", "Simulation._open_output", "output: open"),
    ("fastmace.simulation", "Simulation._write_metadata", "output: metadata"),
    ("fastmace.simulation", "Simulation.run", "Simulation.run"),
    ("fastmace.planner.minimal", "detect_software", "plan: detect_software"),
    ("fastmace.planner.minimal", "MinimalPlanner.hardware", "plan: hardware"),
    ("fastmace.planner.minimal", "MinimalPlanner.plan", "plan: MinimalPlanner.plan"),
    ("fastmace.planner.minimal", "MinimalPlanner.revise_memory", "plan: revise_memory"),
    ("fastmace.planner.minimal", "MinimalPlanner.build", "build: MinimalPlanner.build"),
    ("fastmace.planner.minimal", "MinimalPlanner._execution_model", "build: execution model"),
    ("fastmace.model.forcefield", "ForceField.__init__", "ForceField.__init__"),
    ("fastmace.model.forcefield", "ForceField.execution_model", "build: ForceField.execution_model"),
    ("fastmace.model.templates", "TemplateCache.execution_model", "build: template cache"),
    ("fastmace.model.templates", "private", "build: templates.private (copy)"),
    ("fastmace.model.forces", "MACEForceField.__init__", "build: MACEForceField.__init__"),
    ("fastmace.model.forces", "MACEForceField._setup_species", "build: species setup"),
    ("fastmace.model.forces", "MACEForceField._setup_tables", "build: species tables"),
    ("fastmace.rewrites.registry", "Registry.resolve", "rewrites: resolve"),
    ("fastmace.rewrites.registry", "Registry.apply", "rewrites: apply (incl. self-checks)"),
    ("fastmace.rewrites.registry", "Registry.selfcheck", "rewrites: self-checks"),
    ("fastmace.rewrites.radial", "fit", "radial tables: fit"),
    ("fastmace.rewrites.radial", "certify", "radial tables: certify"),
    ("fastmace.kernels.oeq_streaming.runtime", "render_kernels", "OEq streaming: render"),
    ("fastmace.kernels.oeq_streaming.runtime", "compile_cubin", "OEq streaming: NVRTC compile_cubin"),
    ("fastmace.kernels.oeq_streaming.fused_forward", "render", "fused forward: render"),
    ("fastmace.model.compile", "private_copy", "compile: private_copy"),
    ("fastmace.model.compile", "simplify_traceable", "compile: simplify_traceable"),
    ("fastmace.model.compile", "compile_model", "compile: compile_model (wrap; compiles lazily)"),
    ("fastmace.engine.graph", "GraphEngine.__init__", "engine: GraphEngine.__init__"),
    ("fastmace.engine.graph", "GraphEngine.prepare", "engine: prepare (captures)"),
    ("fastmace.engine.graph", "GraphEngine._initial_forces", "engine: first compiled force call"),
    ("fastmace.engine.graph", "GraphEngine._rebuild", "engine: neighbour-list rebuild"),
    ("fastmace.engine.graph", "GraphEngine._warm_up", "engine: eager warm-up"),
    ("fastmace.engine.graph", "GraphEngine._capture", "engine: graph capture"),
    ("fastmace.engine.constraints", "check_engine", "build: check_engine"),
    ("fastmace.state", "SimulationState.from_atoms", "build: SimulationState.from_atoms"),
    ("fastmace.provenance", "collect", "provenance: collect"),
]

RECORDS = []  # (label, parent label, start s since PROCESS_T0, seconds, depth)
_STACK = []
FIRST_STEP = {}


def _sync():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized() and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()


def timed(owner, attribute, label):
    original = owner.__dict__.get(attribute) if isinstance(owner, type) else getattr(owner, attribute, None)
    if original is None or getattr(original, "_fp32_timed", False):
        return False
    is_static = isinstance(original, staticmethod)
    function = original.__func__ if isinstance(original, (staticmethod, classmethod)) else original

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        _sync()
        parent = _STACK[-1] if _STACK else ""
        _STACK.append(label)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            _sync()
            seconds = time.perf_counter() - started
            _STACK.pop()
            RECORDS.append((label, parent, started - PROCESS_T0, seconds, len(_STACK)))

    wrapper._fp32_timed = True
    setattr(owner, attribute, staticmethod(wrapper) if is_static else
            classmethod(wrapper) if isinstance(original, classmethod) else wrapper)
    return True


def instrument():
    wrapped = {}
    for module_name, path, label in TARGETS:
        try:
            owner = importlib.import_module(module_name)
            *owners, attribute = path.split(".")
            for name in owners:
                owner = getattr(owner, name)
            wrapped[label] = timed(owner, attribute, label)
        except (ImportError, AttributeError) as error:
            wrapped[label] = f"missing: {error}"
    # fm.md reaches md through the package namespace: wrap that binding too.
    if fastmace.__dict__.get("md") is not None and not getattr(fastmace.md, "_fp32_timed", False):
        fastmace.md = importlib.import_module("fastmace.quick").md
    return wrapped


def first_step_timer():
    """Time of the end of the engine's first step (since PROCESS_T0) and that step's own wall time."""
    from fastmace.engine import graph

    original = graph.GraphEngine.step

    def step(self):
        if "end_s" in FIRST_STEP:
            return original(self)
        _sync()
        started = time.perf_counter()
        out = original(self)
        self.flush()
        _sync()
        FIRST_STEP.update(step_s=time.perf_counter() - started, end_s=time.perf_counter() - PROCESS_T0)
        return out

    graph.GraphEngine.step = step


def summary(records):
    totals = defaultdict(lambda: dict(total_s=0.0, calls=0, parents=set()))
    for label, parent, _start, seconds, _depth in records:
        totals[label]["total_s"] += seconds
        totals[label]["calls"] += 1
        totals[label]["parents"].add(parent)
    return {k: dict(total_s=v["total_s"], calls=v["calls"], parents=sorted(v["parents"]))
            for k, v in sorted(totals.items(), key=lambda item: -item[1]["total_s"])}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--timed-steps", type=int, default=1000, help="Steady state: the last N steps of a second run()")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--label", default="", help="cold or warm (recorded)")
    p.add_argument("--cprofile", action="store_true")
    p.add_argument("--allow-busy-gpu", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import ab_step_time

    started = time.perf_counter()
    import torch

    import_torch_s = time.perf_counter() - started
    if not args.allow_busy_gpu and ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    wrapped = instrument()
    first_step_timer()
    import numpy as np
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    structure, atoms = ab_step_time.read_structure(args.structure, None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(20261007))
    Stationary(atoms)
    import fastmace as fm

    out = dict(tool="fp32_setup", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
               label=args.label, fastmace_commit=ab_step_time.fastmace_commit(), python=sys.executable,
               environment={key: os.environ.get(key) for key in (
                   "FASTMACE_CACHE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "MACE_COMPILE_CACHE",
                   "TORCHINDUCTOR_COMPILE_THREADS", "FASTMACE_MODELS", "CUDA_VISIBLE_DEVICES")},
               structure=structure, natoms=len(atoms), wrapped=wrapped, import_fastmace_s=IMPORT_FASTMACE_S,
               import_torch_s=import_torch_s, args={k: str(v) if isinstance(v, Path) else v
                                                    for k, v in vars(args).items()})
    profiler = None
    if args.cprofile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    md_started = time.perf_counter()
    result = fm.md(atoms, model=args.model, temperature_K=args.temperature_K, timestep_fs=args.timestep_fs,
                   steps=args.steps, output=args.output / "run", seed=args.seed)
    _sync()
    md_s = time.perf_counter() - md_started
    if profiler is not None:
        profiler.disable()
        import io
        import pstats
        text = io.StringIO()
        pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(150)
        (args.output / "cprofile.txt").write_text(text.getvalue())
    sim = result.simulation
    # Steady state with output: a second run() of --timed-steps steps (graphs captured already).
    _sync()
    t = time.perf_counter()
    sim.run(steps=args.timed_steps)
    sim.flush_output()
    _sync()
    steady_ms = (time.perf_counter() - t) * 1e3 / args.timed_steps
    try:
        from torch._dynamo.utils import compile_times
        dynamo = compile_times(repr="str", aggregate=True)
    except Exception as error:  # noqa: BLE001  (a measurement record, not control flow)
        dynamo = f"unavailable: {error}"
    try:
        from torch._dynamo.utils import counters
        dynamo_counters = {k: dict(v) for k, v in counters.items() if v}
    except Exception as error:  # noqa: BLE001
        dynamo_counters = f"unavailable: {error}"
    md_start_s = md_started - PROCESS_T0
    out.update(
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        md_call_start_s=md_start_s, md_total_s=md_s, first_step=FIRST_STEP,
        setup_to_first_step_s=(FIRST_STEP["end_s"] - md_start_s) if "end_s" in FIRST_STEP else None,
        process_to_first_step_s=FIRST_STEP.get("end_s"),
        simulation_setup_s=dict(getattr(sim, "_setup_s", {}) or {}),
        steady_ms_per_step_with_output=steady_ms, steps=args.steps,
        plan=str(sim.plan.describe()) if sim.plan is not None else None,
        engine_info=dict(sim._engine.info()) if getattr(sim, "_engine", None) is not None else None,
        stages=summary([r for r in RECORDS if FIRST_STEP.get("end_s") is None or r[2] < FIRST_STEP["end_s"]]),
        stages_after_first_step=summary([r for r in RECORDS if FIRST_STEP.get("end_s") is not None
                                         and r[2] >= FIRST_STEP["end_s"]]),
        timeline=[dict(label=r[0], parent=r[1], start_s=round(r[2], 4), seconds=round(r[3], 4), depth=r[4])
                  for r in sorted(RECORDS, key=lambda r: r[2])],
        dynamo_compile_times=dynamo, dynamo_counters=dynamo_counters)
    (args.output / "setup.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("structure", "label", "setup_to_first_step_s", "process_to_first_step_s",
                                          "steady_ms_per_step_with_output", "simulation_setup_s")},
                     indent=1, default=str))
    for label, v in list(out["stages"].items())[:40]:
        print(f"  {v['total_s']:8.3f} s  x{v['calls']:<4} {label}  <- {', '.join(v['parents'])}")
    return out


if __name__ == "__main__":
    main()
