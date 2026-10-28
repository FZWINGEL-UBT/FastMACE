"""Same-process A/B of two rewrite sets of FastMACE (speed screens, plan task 2-T20; plan opt-1).

Sides "a" and "b" are engines from the minimal planner for the same structure and model, MH-1 hybrid float32
NVT by default, that differ only in the ``RewriteConfig`` fields given (``--a FIELD=VALUE ...`` and
``--b FIELD=VALUE ...``; every other field stays auto). Rewrite records change only modules of their own
model copy, so two sides of one FastMACE version share no process-wide state. Both sides warm up outside
the compile-counter window; then their timed blocks alternate (``fastmace.bench.ab.run_ab``, ABAB or ABBA),
and the result is the ratio of medians b/a with the per-pair ratios. A run is invalid when anything
compiled while timed or another process used the GPU. Writes ``<output>.json``. Use an idle GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402


def rewrite_overrides(items):
    """{field: value} of ``FIELD=VALUE`` items, each a RewriteConfig field; "on"/"off"/"auto" or true/false."""
    from fastmace.config import RewriteConfig

    fields = {f.name: f for f in dataclasses.fields(RewriteConfig)}
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in fields or not value:
            raise ValueError(f"{item!r}: expected FIELD=VALUE with a RewriteConfig field")
        if isinstance(fields[name].default, bool):
            if value.lower() not in ("true", "false"):
                raise ValueError(f"{item!r}: {name} takes true or false")
            out[name] = value.lower() == "true"
        else:
            out[name] = value
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--a", nargs="*", default=[], metavar="FIELD=VALUE", help="Side a's RewriteConfig fields")
    p.add_argument("--b", nargs="*", default=[], metavar="FIELD=VALUE", help="Side b's RewriteConfig fields")
    p.add_argument("--a-engine", nargs="*", default=[], metavar="FIELD=true|false",
                   help="Side a's boolean EngineConfig fields (opt-2 OPT2-T04: overlap_rebuild, step_graph)")
    p.add_argument("--b-engine", nargs="*", default=[], metavar="FIELD=true|false", help="Side b's, as --a-engine")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"])
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--pattern", choices=("abab", "abba"), default="abab")
    p.add_argument("--order", choices=("ab", "ba"), default="ab", help="Build order")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32",
                   help="Both sides' force precision (float64: the decision 63 package by default)")
    args = p.parse_args(argv)
    try:
        args.a, args.b = rewrite_overrides(args.a), rewrite_overrides(args.b)
        args.a_engine, args.b_engine = engine_overrides(args.a_engine), engine_overrides(args.b_engine)
    except ValueError as exc:
        p.error(str(exc))
    if args.a == args.b and args.a_engine == args.b_engine:
        p.error("the two sides have the same rewrite and engine fields")
    return args


def engine_overrides(items):
    """{field: bool} of ``FIELD=true|false`` items, each a boolean EngineConfig field."""
    from fastmace.config import EngineConfig

    fields = {f.name: f for f in dataclasses.fields(EngineConfig) if isinstance(f.default, bool)}
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in fields or value.lower() not in ("true", "false"):
            raise ValueError(f"{item!r}: expected FIELD=true|false with a boolean EngineConfig field")
        out[name] = value.lower() == "true"
    return out


def main(argv=None):
    args = parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from torch._dynamo.utils import counters

    from fastmace.bench import ab, step_time
    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    planners = {"a": minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid", **args.a_engine)),
                "b": minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid", **args.b_engine))}
    hardware = planners["a"].hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    fields = {"a": args.a, "b": args.b}
    order = list(args.order)
    meta = dict(tool="ab_rewrites", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, checkpoint_sha256=sha,
                sides=fields, engine_sides={"a": args.a_engine, "b": args.b_engine},
                protocol=protocol.as_dict(), pattern=args.pattern, order=order, precision=args.precision,
                load=list(os.getloadavg()), structures={})
    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        sides, setup, lines = {}, {}, {}
        for side in order:
            started = time.perf_counter()
            rewrites = RewriteConfig(**fields[side]) if fields[side] else "auto"
            field = ForceField(loaded, base, precision=args.precision, rewrites=rewrites)
            integrator = IntegratorConfig(temperature_K=args.temperature_K)
            planner = planners[side]
            plan = planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
            engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K)
            sides[side] = ab_step_time.FastMACESide(engine, torch)
            setup[side] = time.perf_counter() - started
            lines[side] = [line.strip() for line in plan.describe(full=True).splitlines()
                           if any(f"RewriteConfig.{f} " in line for f in {**args.a, **args.b})]
        for side in order:  # warm up outside the compile-counter window (new edge sizes compile here)
            step_time.warm_up(sides[side], protocol.warmup)
        counted = ab_step_time.compile_counters(counters)
        result = ab.run_ab(sides, dataclasses.replace(protocol, warmup=0), pattern=args.pattern)
        changes, recompiled = ab_step_time.compile_changes(counted, ab_step_time.compile_counters(counters))
        medians = {side: result.statistics[side].median_ms for side in sides}
        blocks = {side: result.statistics[side].block_ms for side in sides}
        entry = dict(natoms=len(base), ab=result.as_dict(), ratio_b_over_a=medians["b"] / medians["a"],
                     pair_ratios=[b / a for a, b in zip(blocks["a"], blocks["b"])], setup_s=setup, plan_lines=lines,
                     compile_during_timing=dict(changes=changes, recompiled=recompiled),
                     observables={side: sides[side].observables() for side in sides}, load_after=list(os.getloadavg()),
                     other_gpu_processes=[pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()])
        entry["valid"] = not recompiled and not entry["other_gpu_processes"]
        meta["structures"][name] = entry
        print(name, " ".join(f"{s} {medians[s]:.4f} ms" for s in sides), f"b/a {entry['ratio_b_over_a']:.4f}",
              "valid" if entry["valid"] else "INVALID", flush=True)
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        sides = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
