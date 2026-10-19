"""1b-T11 probe: where each engine's setup time goes, in a single-engine process (--side).

Wraps timers around the stages of one engine's construction and warm-up, without changing what runs:

- fastmace: planner.load (sha256, OEq/cuEq checks, MACE conversion), planner.plan, planner.build
  split into registry.apply (each record's apply and self-check timed separately), private_copy,
  simplify_traceable, the initial force evaluation (first compiled call: dynamo + Inductor), the
  rest of build; then warm-up split into captures (each _capture timed) and plain steps.
- baseline: workflow.load_md_calculator, GPUEngine split into compile_model's deepcopy +
  generic_rewrites + block_linears + merge_residual_linears + _simplify_traceable, the initial
  force evaluation, the rest; then warm-up with each _capture timed.

``--check off`` builds FastMACE without its self-checks (FixedPlanner.build(check=False)), to
measure them by difference too. Output: <output>/setup.json. Run each configuration twice when
the Inductor cache should be warm (the second process then measures a warm-cache setup).
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

TOOLS = Path(fastmace.__file__).resolve().parents[1] / "tools"
TIMES = defaultdict(list)  # stage -> [seconds, ...]


def _ab_tool():
    spec = importlib.util.spec_from_file_location("_ab_step_time", TOOLS / "ab_step_time.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(owner, attribute, stage, sync=True):
    """Replace owner.attribute by a wrapper that adds its wall time (CUDA-synchronised) to TIMES[stage]."""
    original = getattr(owner, attribute)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        import torch
        if sync and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            if sync and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            label = stage(*args, **kwargs) if callable(stage) else stage
            TIMES[label].append(time.perf_counter() - started)

    setattr(owner, attribute, wrapper)
    return original


def summary():
    return {stage: dict(total_s=sum(v), calls=len(v), each_s=v) for stage, v in TIMES.items()}


def _kind(key):
    return key[0] if isinstance(key, tuple) else key


def instrument_fastmace():
    from fastmace.engine import graph
    from fastmace.model import compile as compiling
    from fastmace.model import forces, loading
    from fastmace.planner import fixed
    from fastmace.rewrites import registry as reg

    timed(loading, "sha256_of", "load: sha256")
    timed(loading, "check_oeq", "load: check_oeq (OEq extension)")
    timed(loading, "check_cueq", "load: check_cueq")
    timed(loading, "require_fused_cueq", "load: require_fused_cueq")
    timed(fixed.FixedPlanner, "plan", "plan")
    timed(reg.Registry, "selfcheck", lambda self, name, *a, **k: f"selfcheck: {name}")
    original_apply = reg.Registry.apply

    def apply(self, model, resolution, context=None, original=None, check=True):
        # Time each record's own apply separately from its self-check (the selfcheck wrapper above).
        records = self._records
        saved = dict(records)
        try:
            for name in resolution.enabled:
                record = records[name]

                def run(model, context, _apply=record.apply, _name=name):
                    import torch
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    try:
                        return _apply(model, context)
                    finally:
                        torch.cuda.synchronize()
                        TIMES[f"apply: {_name}"].append(time.perf_counter() - started)

                import dataclasses
                records[name] = dataclasses.replace(record, apply=run)
            return original_apply(self, model, resolution, context, original, check)
        finally:
            records.clear()
            records.update(saved)

    reg.Registry.apply = apply
    timed(compiling, "private_copy", "compile: private_copy (deepcopy)")
    timed(compiling, "simplify_traceable", "compile: simplify_traceable")
    timed(forces.MACEForceField, "__init__", "build: force field (total)")
    timed(graph.GraphEngine, "_initial_forces", "build: initial forces (first compiled call)")
    timed(graph.GraphEngine, "_capture", lambda self, key: f"warm-up: capture {_kind(key)}")
    timed(graph.GraphEngine, "_warm_up", lambda self, key: f"warm-up: eager warm-up {_kind(key)}")


def instrument_baseline():
    from langevin import gpu_langevin
    timed(gpu_langevin, "compile_model", "compile_model (total)")
    timed(gpu_langevin, "generic_rewrites", "rewrites: generic_rewrites")
    timed(gpu_langevin, "block_linears", "rewrites: block_linears")
    timed(gpu_langevin, "merge_residual_linears", "rewrites: merge_residual_linears")
    timed(gpu_langevin, "_simplify_traceable", "compile: simplify_traceable")
    timed(gpu_langevin, "_restore_oeq_wrappers", "compile: restore OEq wrappers")
    timed(gpu_langevin.MACEForces, "__init__", "build: force field (total)")
    timed(gpu_langevin.GPULangevin, "_initial_forces", "build: initial forces (first compiled call)")
    timed(gpu_langevin.GPULangevin, "_capture",
          lambda self, key: f"warm-up: capture {key[0] if isinstance(key, tuple) else key}")
    import copy
    original_deepcopy = copy.deepcopy

    def deepcopy(value, memo=None, _nil=[]):  # noqa: B006 (copy.deepcopy's own signature)
        import torch
        if memo is None and isinstance(value, torch.nn.Module):
            started = time.perf_counter()
            try:
                return original_deepcopy(value)
            finally:
                TIMES["compile: deepcopy (module)"].append(time.perf_counter() - started)
        return original_deepcopy(value, memo, _nil)

    gpu_langevin.copy = type(copy)("copy_timed")
    gpu_langevin.copy.deepcopy = deepcopy
    gpu_langevin.copy.copy = copy.copy


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--side", choices=("baseline", "fastmace"), required=True)
    p.add_argument("--model", default="mp0-medium")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--structures", nargs="+", default=["NaCl"])
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--check", choices=("on", "off"), default="on")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature-K", type=float, default=300.0)
    args = p.parse_args(argv)
    process_started = time.perf_counter()

    tool = _ab_tool()
    ab_args = tool.parse_args(["--output", str(args.output / "unused"), "--model", args.model, "--checkpoint",
                               str(args.checkpoint), "--sides", args.side, "--seed", str(args.seed)]
                              + (["--baseline", str(args.baseline)] if args.baseline else []))
    from fastmace.bench import ab, step_time

    experiment = None
    stages = {}
    if args.side == "fastmace":
        started = time.perf_counter()
        oeq_runtime = tool.prepare_fastmace_hybrid()
        stages["prepare_hybrid (OEq runtime env)"] = time.perf_counter() - started
    else:
        oeq_runtime = None
        export = ab.load_baseline(args.baseline)
        ab.activate(export)
        from langevin import compile_cache
        from langevin import workflow as experiment
    if tool.gpu_processes():
        raise RuntimeError("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.side == "baseline":
        experiment.require_hybrid_wrapper()
        compile_cache.configure(args.output)
        from langevin import gpu_langevin
        gpu_langevin.configure_allocator()
        instrument_baseline()
    else:
        instrument_fastmace()
        planner = tool.fastmace_planner(ab_args)
        hardware = planner.hardware()
    import torch

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename, python=sys.executable,
               fastmace_commit=tool.fastmace_commit(), baseline=str(args.baseline), model=args.model,
               side=args.side, check=args.check, oeq_runtime=oeq_runtime, structures={},
               inductor_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"))
    started = time.perf_counter()
    if args.side == "fastmace":
        loaded = planner.load(args.model, checkpoint=str(checkpoint))
    else:
        import copy
        md = experiment.parse_args(["--model", args.model, "--checkpoint", str(checkpoint), "--backend", "hybrid"])
        experiment.apply_polar_options(md)
        common, calc = experiment.load_md_calculator(md.dtype, True, "cuda:0", checkpoint, label=args.model,
                                                     cueq_conv_fusion=None)
        experiment.configure_md_calculator(calc, compute_stress=False)
        options = copy.copy(md)
        experiment.adapt_options_to_model(options, calc.models[0])
        flags = {flag: getattr(options, flag) for flag in tool.FLAGS}
        torch.set_default_dtype(torch.float32)
    stages["load (checkpoint, conversion)"] = time.perf_counter() - started
    out["process"] = dict(stages=stages, load_stages=summary())
    TIMES.clear()
    from fastmace.model.compile import CompiledModelCache
    cache = CompiledModelCache()
    for spec in args.structures:
        structure, base = tool.read_structure(spec, experiment)
        TIMES.clear()
        entry = out["structures"][structure] = dict(natoms=len(base))
        started = time.perf_counter()
        if args.side == "fastmace":
            plan = planner.plan(loaded, base, tool.fastmace_integrator(ab_args), hardware)
            engine = planner.build(plan, loaded, base.copy(), temperature_K=args.temperature_K, compile_cache=cache,
                                   check=args.check == "on")
            side = tool.FastMACESide(engine, torch)
        else:
            engine = experiment.GPUEngine(base.copy(), calc, timestep_fs=md.timestep_fs, friction=md.friction,
                                          seed=args.seed, skin=md.skin, compile_mode=md.compile_mode,
                                          device="cuda:0", ensemble="nvt", cell_mode="hydrostatic",
                                          **{key: getattr(md, key) for key in tool.NPT_KEYWORDS}, **flags)
            side = tool.ResearchEngineSide(engine, args.temperature_K, torch)
        torch.cuda.synchronize()
        entry["setup_s"] = time.perf_counter() - started
        entry["setup_stages"] = summary()
        TIMES.clear()
        started = time.perf_counter()
        step_time.warm_up(side, args.warmup)
        torch.cuda.synchronize()
        entry["warmup_s"] = time.perf_counter() - started
        entry["warmup_stages"] = summary()
        stats = step_time.time_engine(side, step_time.TimingProtocol(warmup=0, block=300, blocks=2))
        entry["step_ms"] = stats.median_ms
        entry["info"] = dict(engine.info()) if args.side == "fastmace" else {
            k: v for k, v in engine.dyn.__dict__.items() if k in ("capture_memory",)}
        if args.side == "fastmace":
            entry["force_field"] = dict(engine.force_field.info())
        side = engine = None
        torch.cuda.empty_cache()
    out["process_wall_s"] = time.perf_counter() - process_started
    (args.output / "setup.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for structure, entry in out["structures"].items():
        print(structure, f"setup {entry['setup_s']:.2f} s warm-up {entry['warmup_s']:.2f} s "
                         f"step {entry['step_ms']:.3f} ms")
        for stage, value in sorted(entry["setup_stages"].items(), key=lambda kv: -kv[1]["total_s"]):
            print(f"  {stage}: {value['total_s']:.3f} s ({value['calls']})")
        for stage, value in sorted(entry["warmup_stages"].items(), key=lambda kv: -kv[1]["total_s"]):
            print(f"  {stage}: {value['total_s']:.3f} s ({value['calls']})")


if __name__ == "__main__":
    main()
