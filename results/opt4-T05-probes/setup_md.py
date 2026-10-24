"""OPT4-T05 probe: one ``fm.md`` start, split into stages, without changing what runs.

2-T19's method (``results/2-T19-probes/setup_stages.py``): CUDA-synchronised timers are wrapped around the
functions of one ``fm.md`` call, from the process start to its return, and the stages are summed into
import, load, self-checks, compile, capture and the rest. One structure per process; the cache state
(cold: a fresh ``FASTMACE_CACHE``; warm: a filled one) is the caller's.

Not library code. Output: <output>/setup.json and the fm.md output directory <output>/md.
"""

from __future__ import annotations

import time

PROCESS_T0 = time.perf_counter()

import argparse  # noqa: E402
import functools  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

started = time.perf_counter()
import fastmace as fm  # noqa: E402  (before torch: soft file locks, SPEC §14)

IMPORT_FASTMACE_S = time.perf_counter() - started
started = time.perf_counter()
import torch  # noqa: E402

IMPORT_TORCH_S = time.perf_counter() - started

TIMES = defaultdict(list)  # stage -> [seconds, ...]
STACK = []  # the stages open now: a stage's self time excludes the stages timed inside it
SELF = defaultdict(float)  # stage -> seconds not inside another timed stage


def _sync():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _since_process_start():
    """Seconds since this process started (Linux: /proc/self/stat field 22 against /proc/uptime)."""
    with open("/proc/self/stat") as handle:
        fields = handle.read().rsplit(")", 1)[1].split()
    with open("/proc/uptime") as handle:
        uptime = float(handle.read().split()[0])
    return uptime - int(fields[19]) / os.sysconf("SC_CLK_TCK")


def timed(owner, attribute, stage):
    """Replace owner.attribute by a wrapper that adds its wall time (CUDA-synchronised) to TIMES[stage]."""
    original = getattr(owner, attribute, None)
    if original is None:
        return False

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        label = stage(*args, **kwargs) if callable(stage) else stage
        _sync()
        begin = time.perf_counter()
        STACK.append([label, 0.0])
        try:
            return original(*args, **kwargs)
        finally:
            _sync()
            seconds = time.perf_counter() - begin
            _, inner = STACK.pop()
            TIMES[label].append(seconds)
            SELF[label] += seconds - inner
            if STACK:
                STACK[-1][1] += seconds

    if isinstance(owner, type) and isinstance(owner.__dict__.get(attribute), staticmethod):
        wrapper = staticmethod(wrapper)
    setattr(owner, attribute, wrapper)
    return True


def _kind(key):
    return key[0] if isinstance(key, tuple) else key


def instrument():
    def module(path):
        return importlib.import_module(f"fastmace.{path}")

    loading, minimal, fixed = module("model.loading"), module("planner.minimal"), module("planner.fixed")
    registry, compiling, forces = module("rewrites.registry"), module("model.compile"), module("model.forces")
    graph, forcefield, selfcheck = module("engine.graph"), module("model.forcefield"), module("planner.selfcheck")
    simulation, provenance, quick = module("simulation"), module("provenance"), module("quick")
    wrapped = {}

    def wrap(owner, attribute, stage):
        wrapped[f"{getattr(owner, '__name__', owner)}.{attribute}"] = timed(owner, attribute, stage)

    wrap(loading, "prepare_hybrid", "load: prepare_hybrid (OEq runtime)")
    wrap(loading, "sha256_of", "load: sha256")
    wrap(loading, "check_oeq", "load: check_oeq")
    wrap(loading, "check_cueq", "load: check_cueq")
    wrap(loading, "load", "load: loading.load")
    wrap(loading, "load_pristine", "load: load_pristine")
    wrap(torch, "load", "load: torch.load")
    wrap(forcefield.ForceField, "__init__", "load: ForceField.__init__")
    wrap(forcefield.ForceField, "execution_model", "build: execution model (template / conversion)")
    wrap(minimal.MinimalPlanner, "plan", "plan: MinimalPlanner.plan")
    wrap(minimal.MinimalPlanner, "build", "build: MinimalPlanner.build")
    wrap(fixed.FixedPlanner, "build", "build: FixedPlanner.build")
    wrap(registry.Registry, "resolve", "rewrites: resolve")
    wrap(registry.Registry, "apply", "rewrites: apply")
    wrap(registry.Registry, "selfcheck", lambda self, name, *a, **k: f"self-check: {name}")
    wrap(compiling, "private_copy", "build: private_copy")
    wrap(compiling, "simplify_traceable", "build: simplify_traceable")
    wrap(forces.MACEForceField, "__init__", "build: MACEForceField.__init__")
    wrap(graph.GraphEngine, "__init__", "build: GraphEngine.__init__")
    wrap(graph.GraphEngine, "_initial_forces", "compile: first compiled call")
    wrap(graph.GraphEngine, "prepare", "capture: prepare (setup)")
    wrap(graph.GraphEngine, "_capture", lambda self, key: f"capture: {_kind(key)}")
    wrap(graph.GraphEngine, "_warm_up", lambda self, key: f"capture: eager warm-up {_kind(key)}")
    wrap(selfcheck, "references", "gate: references")
    wrap(selfcheck, "production", "gate: production")
    wrap(provenance, "fastmace_block", "rest: provenance block")
    wrap(simulation.Simulation, "__init__", "simulation: __init__")
    wrap(simulation.Simulation, "run", "run: Simulation.run")
    wrap(simulation.Simulation, "_write_metadata", "rest: metadata.json")
    wrap(simulation.Simulation, "_open_output", "rest: open output")
    wrap(quick, "md", "fm.md")
    return wrapped


def compile_times():
    try:
        from torch._dynamo.utils import compile_times as times
        return times(repr="str", aggregate=True)
    except Exception as error:  # pragma: no cover - torch version
        return f"unavailable: {error}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--check", default="skip")
    p.add_argument("--label", default="", help="free text recorded in setup.json (cache state)")
    p.add_argument("--dynamic-edges", action="store_true",
                   help="fm.md(engine=EngineConfig(dynamic_edges=True)): edges marked dynamic from the first call")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    tools = Path(fm.__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    import ab_step_time

    structure, atoms = ab_step_time.read_structure(args.structure, None)
    wrapped = instrument()
    out = dict(schema=1, tool="opt4-T05 setup_md", created_utc=datetime.now(timezone.utc).isoformat(),
               host=os.uname().nodename, python=sys.executable, fastmace=fm.__file__,
               fastmace_commit=ab_step_time.fastmace_commit(), torch=torch.__version__, label=args.label,
               model=args.model, structure=structure, natoms=len(atoms), steps=args.steps, check=args.check,
               environment={key: os.environ.get(key) for key in (
                   "FASTMACE_CACHE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_COMPILE_THREADS",
                   "XDG_CACHE_HOME", "CUDA_VISIBLE_DEVICES")},
               wrapped=wrapped, import_fastmace_s=IMPORT_FASTMACE_S, import_torch_s=IMPORT_TORCH_S)
    path = args.output / "setup.json"
    before_md = time.perf_counter()
    out["before_md_s"] = before_md - PROCESS_T0
    out["process_start_to_main_s"] = _since_process_start() - (time.perf_counter() - PROCESS_T0)
    extra = {}
    if args.dynamic_edges:
        from fastmace.config import EngineConfig

        extra["engine"] = EngineConfig(dynamic_edges=True)
    out["dynamic_edges"] = args.dynamic_edges
    result = fm.md(atoms, model=args.model, temperature_K=args.temperature_K, steps=args.steps, seed=args.seed,
                   check=args.check, output=args.output / "md", **extra)
    _sync()
    out["md_s"] = time.perf_counter() - before_md
    out["since_process_start_s"] = _since_process_start()
    out["gpu"] = torch.cuda.get_device_name(0)
    out["stages"] = {stage: dict(total_s=sum(v), self_s=SELF[stage], calls=len(v), each_s=v)
                     for stage, v in TIMES.items()}
    sim = result.simulation
    out["simulation_setup_s"] = dict(getattr(sim, "_setup_s", {}) or {})
    out["plan"] = result.plan.describe() if hasattr(result.plan, "describe") else repr(result.plan)
    engine = getattr(sim, "_engine", None)
    if engine is not None:
        import numpy as np

        state = engine.state
        np.savez(args.output / "final_state.npz", positions=state.positions.detach().cpu().numpy(),
                 velocities=state.velocities.detach().cpu().numpy(),
                 forces=state.forces.detach().cpu().numpy(), energy=state.energy.detach().cpu().numpy(),
                 step=int(state.step))
        out["engine_config"] = repr(getattr(engine, "config", None))
        out["engine"] = dict(capture_count=getattr(engine, "capture_count", None),
                             capture_seconds=getattr(engine, "capture_seconds", None),
                             recapture_log=getattr(engine, "recapture_log", None))
    out["summary"] = result.summary
    from fastmace.kernels.triton import cache

    out["cache_stats"] = cache.cache_stats()
    out["compile_times"] = compile_times()
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"{structure} {args.model} md {out['md_s']:.2f} s, process {out['since_process_start_s']:.2f} s")
    for stage, value in sorted(out["stages"].items(), key=lambda kv: -kv[1]["self_s"])[:30]:
        print(f"  {stage}: self {value['self_s']:.3f} s, total {value['total_s']:.3f} s ({value['calls']})")


if __name__ == "__main__":
    main()
