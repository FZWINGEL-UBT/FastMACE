"""2-T19 probe: setup time per stage, FastMACE 1b (main) against the 1a release, one version per process.

Wraps timers (CUDA-synchronised) around the stages of one side's setup without changing what runs, as
1b-T11's ``setup_probe.py`` did, on the two paths 1b-T23 measured:

- ``--path ab`` (``--side 1a`` or ``1b``): ``tools/ab_fastmace.py``'s own load and build of that side, so
  "setup" is exactly 1b-T23's ``setup_s`` and "load" its ``load_s``. 1a converts the checkpoint in its
  load; 1b loads the pristine model there and converts it per plan, inside the build.
- ``--path simulation`` (1b only): ``fastmace.Simulation`` with ``--check`` (``gate`` or ``skip``), whose own
  setup record (plan, build, gate, provenance) is kept beside the stage timers.

Stages: load (sha256, MACE calculator = torch.load plus conversion, cuEq/OEq checks), plan, execution model
(1b: reload, hash and conversion per plan), force field, rewrites (each record's apply, each self-check),
private copy, first compiled call, gate (references, production), then warm-up (captures, eager warm-ups).

Not library code. Output: <output>/setup.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

REPOSITORY = Path(fastmace.__file__).resolve().parents[1]
TOOLS = REPOSITORY / "tools"
TIMES = defaultdict(list)  # stage -> [seconds, ...]
_DEPTH = [0]  # nesting depth, so that a stage inside another stage is still timed but marked


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    spec.loader.exec_module(module)
    return module


def _sync():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def timed(owner, attribute, stage):
    """Replace owner.attribute by a wrapper that adds its wall time (CUDA-synchronised) to TIMES[stage]."""
    original = getattr(owner, attribute, None)
    if original is None:
        return False
    if getattr(original, "_t19_timed", False):
        return True

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        _sync()
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _sync()
            label = stage(*args, **kwargs) if callable(stage) else stage
            TIMES[label].append(time.perf_counter() - started)

    wrapper._t19_timed = True
    setattr(owner, attribute, wrapper)
    return True


def _kind(key):
    return key[0] if isinstance(key, tuple) else key


def instrument(package):
    """Timers on one FastMACE package (``fastmace`` or ``fastmace_1a``); returns the stages it could wrap."""
    name = package.__name__

    def module(path):
        try:
            return importlib.import_module(f"{name}.{path}")
        except ImportError:
            return None

    loading, fixed, minimal = module("model.loading"), module("planner.fixed"), module("planner.minimal")
    registry, compiling, forces = module("rewrites.registry"), module("model.compile"), module("model.forces")
    graph, forcefield, selfcheck = module("engine.graph"), module("model.forcefield"), module("planner.selfcheck")
    from mace import calculators

    wrapped = {}
    wrapped["load: sha256_of"] = timed(loading, "sha256_of", "load: sha256_of")
    wrapped["load: check_oeq"] = timed(loading, "check_oeq", "load: check_oeq")
    wrapped["load: check_cueq"] = timed(loading, "check_cueq", "load: check_cueq")
    wrapped["load: require_fused_cueq"] = timed(loading, "require_fused_cueq", "load: require_fused_cueq")
    wrapped["load: MACECalculator"] = timed(calculators.MACECalculator, "__init__",
                                            "load: MACECalculator (torch.load + conversion)")
    wrapped["load: loading.load"] = timed(loading, "load", "load: loading.load (total)")
    wrapped["load: load_pristine"] = timed(loading, "load_pristine", "load: load_pristine (total)")
    wrapped["plan: fixed"] = timed(fixed.FixedPlanner, "plan", "plan: FixedPlanner.plan")
    wrapped["build: fixed"] = timed(fixed.FixedPlanner, "build", "build: FixedPlanner.build (total)")
    if minimal is not None:
        wrapped["plan: minimal"] = timed(minimal.MinimalPlanner, "plan", "plan: MinimalPlanner.plan (total)")
        wrapped["build: minimal"] = timed(minimal.MinimalPlanner, "build", "build: MinimalPlanner.build (total)")
        wrapped["build: execution model"] = timed(minimal.MinimalPlanner, "_execution_model",
                                                  "build: execution model (reload, hash, conversion)")
    if forcefield is not None:
        wrapped["ForceField"] = timed(forcefield.ForceField, "__init__", "force field: ForceField.__init__")
    wrapped["MACEForceField"] = timed(forces.MACEForceField, "__init__", "build: MACEForceField.__init__ (total)")
    wrapped["rewrites: apply"] = timed(registry.Registry, "apply", "rewrites: Registry.apply (total)")
    wrapped["rewrites: selfcheck"] = timed(registry.Registry, "selfcheck",
                                           lambda self, record, *a, **k: f"self-check: {record}")
    wrapped["compile: private_copy"] = timed(compiling, "private_copy", "compile: private_copy")
    wrapped["compile: simplify_traceable"] = timed(compiling, "simplify_traceable", "compile: simplify_traceable")
    wrapped["first compiled call"] = timed(graph.GraphEngine, "_initial_forces",
                                           "build: initial forces (first compiled call)")
    wrapped["capture"] = timed(graph.GraphEngine, "_capture", lambda self, key: f"warm-up: capture {_kind(key)}")
    wrapped["eager warm-up"] = timed(graph.GraphEngine, "_warm_up",
                                     lambda self, key: f"warm-up: eager warm-up {_kind(key)}")
    # The rest of FixedPlanner.build (found after the first run: 1b spent 3-6 s there, 1a 0.4 s).
    wrapped["fixed: rewrite_resolution"] = timed(fixed.FixedPlanner, "rewrite_resolution",
                                                 "build: FixedPlanner.rewrite_resolution")
    wrapped["fixed: configure_process"] = timed(fixed.FixedPlanner, "configure_process",
                                                "build: FixedPlanner.configure_process")
    wrapped["registry: default_registry"] = timed(registry, "default_registry", "build: default_registry")
    wrapped["registry: resolve"] = timed(registry.Registry, "resolve", "build: Registry.resolve")
    constraints, state = module("engine.constraints"), module("state")
    if constraints is not None:
        wrapped["engine: check_engine"] = timed(constraints, "check_engine", "build: constraints.check_engine")
    wrapped["state: from_atoms"] = timed(state.SimulationState, "from_atoms", "build: SimulationState.from_atoms")
    wrapped["engine: __init__"] = timed(graph.GraphEngine, "__init__", "build: GraphEngine.__init__ (total)")
    if selfcheck is not None:
        wrapped["gate: references"] = timed(selfcheck, "references", "gate: references")
        wrapped["gate: production"] = timed(selfcheck, "production", "gate: production")
    return wrapped


def _since_process_start():
    """Seconds since this process started (Linux: /proc/self/stat field 22 against /proc/uptime)."""
    with open("/proc/self/stat") as handle:
        fields = handle.read().rsplit(")", 1)[1].split()
    with open("/proc/uptime") as handle:
        uptime = float(handle.read().split()[0])
    return uptime - int(fields[19]) / os.sysconf("SC_CLK_TCK")


def summary():
    return {stage: dict(total_s=sum(v), calls=len(v), each_s=v) for stage, v in TIMES.items()}


def apply_timer(package):
    """Each enabled record's own apply timed apart from its self-check (setup_probe.py's method)."""
    registry = importlib.import_module(f"{package.__name__}.rewrites.registry")
    current = registry.Registry.apply

    def apply(self, model, resolution, context=None, original=None, check=True):
        records = self._records
        saved = dict(records)
        try:
            for name in resolution.enabled:
                record = records[name]

                def run(model, context, _apply=record.apply, _name=name):
                    _sync()
                    started = time.perf_counter()
                    try:
                        return _apply(model, context)
                    finally:
                        _sync()
                        TIMES[f"apply: {_name}"].append(time.perf_counter() - started)

                records[name] = dataclasses.replace(record, apply=run)
            return current(self, model, resolution, context, original, check)
        finally:
            records.clear()
            records.update(saved)

    registry.Registry.apply = apply


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--path", choices=("ab", "simulation"), default="ab")
    p.add_argument("--side", choices=("1a", "1b"), default="1b")
    p.add_argument("--renamed", type=Path, help="Directory holding the exported fastmace_1a package (--side 1a)")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"])
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--check", choices=("gate", "skip"), default="skip", help="--path simulation only")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature-K", type=float, default=300.0)
    return p.parse_args(argv)


def main(argv=None):
    process_started = time.perf_counter()
    args = parse_args(argv)
    if args.path == "simulation" and args.side != "1b":
        raise SystemExit("--path simulation is 1b only")
    ab_fastmace = _tool("ab_fastmace")
    ab_step_time = ab_fastmace.ab_step_time
    from fastmace import _patches
    from fastmace.bench import step_time, versions
    from fastmace.model import loading

    stages = {}
    mace_first = "mace" in sys.modules  # run through mace_first.py: MACE imported before fastmace
    started = time.perf_counter()
    # 1b's OEq runtime, before anything imports MACE (as ab_fastmace); with MACE first the plan falls back.
    oeq_runtime = None if mace_first else str(loading.prepare_hybrid())
    stages["prepare_hybrid"] = time.perf_counter() - started
    package, source = fastmace, None
    if args.side == "1a":
        started = time.perf_counter()
        package, source = versions.load_renamed(args.renamed, ab_fastmace.RENAMED, ab_fastmace.FASTMACE_1A_COMMIT)
        for name in ("config", "planner.fixed", "model.compile", "_patches"):
            importlib.import_module(f"{ab_fastmace.RENAMED}.{name}")
        versions.share_record(_patches, package._patches, "oeq_runtime")
        stages["import 1a"] = time.perf_counter() - started
    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    import mace.calculators  # noqa: F401  (imported by both sides' loads anyway; needed for the timer)
    wrapped = instrument(package)
    apply_timer(package)
    import torch

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    out = dict(schema=1, tool="2-T19 setup_stages", created_utc=datetime.now(timezone.utc).isoformat(),
               host=os.uname().nodename, python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(),
               fastmace_1a=source, mace_first=mace_first, path=args.path, side=args.side, check=args.check,
               model=args.model,
               checkpoint=str(checkpoint), oeq_runtime=oeq_runtime, wrapped=wrapped,
               environment={key: os.environ.get(key) for key in (
                   "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "FASTMACE_CACHE", "MACE_COMPILE_CACHE",
                   "TORCHINDUCTOR_COMPILE_THREADS", "CUDA_VISIBLE_DEVICES")},
               structures={})
    path = args.output / "setup.json"

    def save():
        path.write_text(json.dumps(out, indent=1, default=str) + "\n")

    save()
    patches = versions.ProcessPatches()
    side = None
    if args.path == "ab":
        ab_args = ab_fastmace.parse_args(["--output", str(args.output / "unused"), "--model", args.model,
                                          "--checkpoint", str(checkpoint), "--sides", args.side,
                                          "--seed", str(args.seed), "--temperature-K", str(args.temperature_K)]
                                         + (["--renamed", str(args.renamed)] if args.renamed else []))
        side = ab_fastmace.Version(args.side, ab_args, package)
        patches.enter(args.side)
        side.prepare()
        patches.leave(args.side)
        TIMES.clear()
        patches.enter(args.side)
        started = time.perf_counter()
        out["load"] = side.load(checkpoint)
        _sync()
        stages["load"] = time.perf_counter() - started
        patches.leave(args.side)
    else:
        from fastmace.model import compile as compiling
        compiling.configure_allocator()  # before the load touches CUDA, as a planner's construction does
        TIMES.clear()
        started = time.perf_counter()
        loaded = loading.load_pristine(args.model, checkpoint=str(checkpoint), sha256=ab_fastmace._sha256(checkpoint))
        stages["load"] = time.perf_counter() - started
    out["gpu"] = torch.cuda.get_device_name(0)
    out["torch"] = torch.__version__
    out["process"] = dict(stages=stages, load_stages=summary())
    save()
    for spec in args.structures:
        structure, base = ab_step_time.read_structure(spec, None)
        entry = out["structures"][structure] = dict(natoms=len(base))
        TIMES.clear()
        if args.path == "ab":
            patches.enter(args.side)
            started = time.perf_counter()
            engine, plans = side.build(base)
            _sync()
            entry["setup_s"] = time.perf_counter() - started
            entry["plan"] = plans["describe"]
        else:
            from ase import units

            from fastmace.integrators import Langevin
            from fastmace.model.forcefield import ForceField
            from fastmace.simulation import Simulation

            # tools/ab_frontends.py's integrator: 0.5 fs, ASE friction 0.02.
            integrator = Langevin(timestep_fs=0.5, temperature_K=args.temperature_K, friction_per_fs=0.02 * units.fs)
            started = time.perf_counter()
            atoms = base.copy()
            simulation = Simulation(atoms, ForceField(loaded, atoms), integrator=integrator, seed=args.seed,
                                    check=args.check)
            _sync()
            entry["setup_s"] = time.perf_counter() - started
            entry["since_process_start_s"] = _since_process_start()
            entry["simulation_setup"] = dict(simulation._setup_s)
            engine = simulation._engine
            entry["plan"] = simulation._plan.describe()
        entry["setup_stages"] = summary()
        TIMES.clear()
        started = time.perf_counter()
        step_time.warm_up(ab_step_time.FastMACESide(engine, torch), args.warmup)
        _sync()
        entry["warmup_s"] = time.perf_counter() - started
        entry["warmup_stages"] = summary()
        if args.path == "ab":
            patches.leave(args.side)
        engine = simulation = None
        torch.cuda.empty_cache()
        save()
    out["process_wall_s"] = time.perf_counter() - process_started
    save()
    print(f"{args.side} {args.path} load {stages['load']:.2f} s")
    for structure, entry in out["structures"].items():
        print(structure, f"setup {entry['setup_s']:.2f} s warm-up {entry['warmup_s']:.2f} s")
        for stage, value in sorted(entry["setup_stages"].items(), key=lambda kv: -kv[1]["total_s"])[:25]:
            print(f"  {stage}: {value['total_s']:.3f} s ({value['calls']})")


if __name__ == "__main__":
    main()
