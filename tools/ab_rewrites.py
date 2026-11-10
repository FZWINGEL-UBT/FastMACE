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
    p.add_argument("--a-engine", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="Side a's boolean, float or string EngineConfig fields (opt-2 OPT2-T04: overlap_rebuild, "
                        "step_graph; CAP-T01: force_graphs)")
    p.add_argument("--b-engine", nargs="*", default=[], metavar="FIELD=true|false", help="Side b's, as --a-engine")
    p.add_argument("--a-neighbours", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="Side a's NeighbourConfig fields (e.g. skin=0.66, edge_bucket_fraction=0.025)")
    p.add_argument("--b-neighbours", nargs="*", default=[], metavar="FIELD=VALUE", help="Side b's, as --a-neighbours")
    p.add_argument("--a-constant", nargs="*", default=[], metavar="MODULE:NAME=VALUE",
                   help="Module constants set while side a is built and before each of its steps (screens; e.g. "
                        "fastmace.kernels.oeq_streaming.fused_forward:BLOCKS_PER_SM=4): what it captures again in "
                        "warm-up (new edge sizes) reads them too")
    p.add_argument("--b-constant", nargs="*", default=[], metavar="MODULE:NAME=VALUE", help="Side b's, as --a-constant")
    p.add_argument("--timestep-fs", type=float, default=None, help="Default: IntegratorConfig's (0.5 fs)")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"])
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--pattern", choices=("abab", "abba"), default="abab")
    p.add_argument("--stop-after", type=int, default=None, metavar="BLOCKS",
                   help="End a structure's run after this many blocks per side when b/a is beyond --stop-beyond")
    p.add_argument("--stop-beyond", type=float, default=0.03, help="Early stop when |b/a - 1| exceeds this")
    p.add_argument("--order", choices=("ab", "ba"), default="ab", help="Build order")
    p.add_argument("--a-tf32", action="store_true",
                   help="Side a builds and steps with PyTorch's TF32 matmul mode (TF32 screen); IEEE otherwise")
    p.add_argument("--b-tf32", action="store_true", help="Side b's, as --a-tf32")
    p.add_argument("--a-recapture", action="store_true",
                   help="Side a drops its CUDA graphs after the build and re-captures them in IEEE (the control of "
                        "--a-tf32's re-capture)")
    p.add_argument("--b-recapture", action="store_true", help="Side b's, as --a-recapture")
    p.add_argument("--prepare", action="store_true",
                   help="Capture the first step's stages inside each side's build, with its --a/--b-constant values "
                        "set, as Simulation does (engine.prepare); otherwise they are captured lazily in the warm-up "
                        "(CAP-T05's snapshot A/B)")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32",
                   help="Both sides' force precision (float64: the decision 63 package by default)")
    p.add_argument("--pack", default=None, metavar="'NAME|ITEM ...;NAME|ITEM ...'",
                   help="Several b sides against one side a in this process (setup paid once for a): each entry's "
                        "items (MODULE:NAME=VALUE constants, FIELD=VALUE RewriteConfig fields) are added to side "
                        "b's. Order ab builds a first, then each b in turn (built, timed, freed); order ba builds "
                        "every b first and a last, so all of them are on the GPU at once. Writes "
                        "<output>-<NAME>.json per entry, each in the two-side format with setup_s.a the shared build")
    args = p.parse_args(argv)
    try:
        args.a, args.b = rewrite_overrides(args.a), rewrite_overrides(args.b)
        args.a_engine, args.b_engine = engine_overrides(args.a_engine), engine_overrides(args.b_engine)
    except ValueError as exc:
        p.error(str(exc))
    try:
        args.a_constant, args.b_constant = constant_overrides(args.a_constant), constant_overrides(args.b_constant)
    except ValueError as exc:
        p.error(str(exc))
    try:
        args.a_neighbours, args.b_neighbours = (neighbour_overrides(args.a_neighbours),
                                                neighbour_overrides(args.b_neighbours))
    except ValueError as exc:
        p.error(str(exc))
    try:
        args.pack = pack_entries(args.pack) if args.pack else None
    except ValueError as exc:
        p.error(str(exc))
    same_rest = (args.a_engine == args.b_engine and args.a_neighbours == args.b_neighbours
                 and args.a_tf32 == args.b_tf32 and args.a_recapture == args.b_recapture)
    for name, fields, constants in args.pack or [(None, {}, {})]:
        if same_rest and args.a == {**args.b, **fields} and args.a_constant == {**args.b_constant, **constants}:
            p.error("the two sides have the same rewrite and engine fields" + (f" (pack entry {name})" if name else ""))
    return args


def pack_entries(text):
    """[(name, RewriteConfig fields, constants)] of ``NAME|ITEM ...;NAME|ITEM ...``: each item is a
    ``MODULE:NAME=VALUE`` constant or a ``FIELD=VALUE`` RewriteConfig field, added to side b's own."""
    import re

    entries, names = [], set()
    for entry in filter(None, (part.strip() for part in text.split(";"))):
        name, bar, body = entry.partition("|")
        items = body.split()
        if not bar or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not items:
            raise ValueError(f"{entry!r}: expected NAME|ITEM ... (NAME of letters, digits, '_', '.', '-')")
        if name in names:
            raise ValueError(f"pack entry {name!r} appears twice")
        names.add(name)
        constants = [item for item in items if ":" in item.partition("=")[0]]
        entries.append((name, rewrite_overrides([item for item in items if item not in constants]),
                        constant_overrides(constants)))
    if not entries:
        raise ValueError("--pack has no entries")
    return entries


def constant_overrides(items):
    """{(module, name): value} of ``MODULE:NAME=VALUE`` items; VALUE is a Python literal of the constant's type."""
    import ast
    import importlib
    from types import MappingProxyType

    out = {}
    for item in items:
        target, _, value = item.partition("=")
        module, _, name = target.partition(":")
        if not module or not name or not value:
            raise ValueError(f"{item!r}: expected MODULE:NAME=VALUE")
        current = getattr(importlib.import_module(module), name)
        new = ast.literal_eval(value)
        if isinstance(current, MappingProxyType) and isinstance(new, dict):  # a per-model table: a dict literal
            new = MappingProxyType(new)
        if current is not None and type(new) is not type(current):  # a None default takes any literal
            raise ValueError(f"{item!r}: {name} is {type(current).__name__}, not {type(new).__name__}")
        out[(module, name)] = new
    return out


class _Constants:
    """Sets module constants for one side's build and restores them afterwards."""

    def __init__(self, values):
        self.values, self.saved = values, {}

    def __enter__(self):
        import importlib
        for (module, name), value in self.values.items():
            owner = importlib.import_module(module)
            self.saved[(module, name)] = getattr(owner, name)
            setattr(owner, name, value)

    def __exit__(self, *exc):
        import importlib
        for (module, name), value in self.saved.items():
            setattr(importlib.import_module(module), name, value)


def built_under(values, make_field, make_plan, make_engine, *, prepare=False):
    """``(field, plan, engine)`` of one side, with the module constants ``values`` set over all three steps and
    restored afterwards: the planner takes its label-keyed decisions in ``make_plan``, so a constant such as
    ``fastmace.planner.fixed:DECISION_66_TF32X3_LABELS`` acts only if it is set there too."""
    with _Constants(values):
        field = make_field()
        plan = make_plan(field)
        engine = make_engine(plan, field)
        if prepare:
            engine.prepare()
    return field, plan, engine


class ConstantsSide:
    """A TimedEngine whose module constants are set before every step, as ``Tf32Side`` sets its matmul mode: a
    graph captured again outside the build (a new edge size in warm-up) reads the side's own values, not the other
    side's (segment_gemm tile order, 30 October 2026: side a's re-captures had read side b's TILE_ORDER).
    ``values`` holds every constant either side sets, the defaults for the ones this side leaves alone."""

    def __init__(self, side, values):
        import importlib
        self.side = side
        self.values = [(importlib.import_module(module), name, value) for (module, name), value in values.items()]

    def step(self):
        for owner, name, value in self.values:
            setattr(owner, name, value)
        self.side.step()

    def __getattr__(self, name):
        return getattr(self.side, name)


def engine_overrides(items):
    """{field: value} of ``FIELD=VALUE`` items: a boolean EngineConfig field with true|false, a float one, or a
    string one (a choice such as ``force_graphs=keep``; EngineConfig validates it)."""
    from fastmace.config import EngineConfig

    fields = {f.name: f for f in dataclasses.fields(EngineConfig)
              if isinstance(f.default, (bool, str)) or type(f.default) is float}
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in fields or not value:
            raise ValueError(f"{item!r}: expected FIELD=VALUE with a boolean, float or string EngineConfig field")
        if isinstance(fields[name].default, bool):
            if value.lower() not in ("true", "false"):
                raise ValueError(f"{item!r}: {name} takes true or false")
            out[name] = value.lower() == "true"
        elif type(fields[name].default) is float:
            out[name] = float(value)
        else:
            out[name] = value
    return out


def neighbour_overrides(items):
    """{field: value} of ``FIELD=VALUE`` items, each a NeighbourConfig field; numbers become floats."""
    from fastmace.config import NeighbourConfig

    fields = {f.name for f in dataclasses.fields(NeighbourConfig)}
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in fields or not value:
            raise ValueError(f"{item!r}: expected FIELD=VALUE with a NeighbourConfig field")
        try:
            out[name] = float(value)
        except ValueError:
            out[name] = value
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
    from fastmace.config import NeighbourConfig

    planners = {"a": minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid", **args.a_engine),
                                            neighbours=NeighbourConfig(**args.a_neighbours)),
                "b": minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid", **args.b_engine),
                                            neighbours=NeighbourConfig(**args.b_neighbours))}
    hardware = planners["a"].hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    fields = {"a": args.a, "b": args.b}
    order = list(args.order)
    meta = dict(tool="ab_rewrites", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, checkpoint_sha256=sha,
                sides=fields, engine_sides={"a": args.a_engine, "b": args.b_engine},
                neighbour_sides={"a": args.a_neighbours, "b": args.b_neighbours},
                constant_sides={side: {f"{m}:{n}": v for (m, n), v in values.items()}
                                for side, values in (("a", args.a_constant), ("b", args.b_constant))},
                tf32_sides={"a": args.a_tf32, "b": args.b_tf32},
                recapture_sides={"a": args.a_recapture, "b": args.b_recapture}, timestep_fs=args.timestep_fs,
                protocol=protocol.as_dict(), pattern=args.pattern, order=order, precision=args.precision,
                prepare=args.prepare, stop_after=args.stop_after, stop_beyond=args.stop_beyond,
                load=list(os.getloadavg()), structures={})
    import importlib
    specs = [(None, args.b, args.b_constant)] if not args.pack else [
        (name, {**args.b, **fields}, {**args.b_constant, **constants}) for name, fields, constants in args.pack]
    every_constant = {**args.a_constant}
    for _, _, constants in specs:
        every_constant.update(constants)
    defaults = {key: getattr(importlib.import_module(key[0]), key[1]) for key in every_constant}
    metas = {}
    for name, fields_b, constants_b in specs:
        entry_meta = dict(meta, sides={"a": args.a, "b": fields_b},
                          constant_sides={side: {f"{m}:{n}": v for (m, n), v in values.items()}
                                          for side, values in (("a", args.a_constant), ("b", constants_b))},
                          structures={})
        if name is not None:
            entry_meta["pack"] = dict(name=name, entries=[entry[0] for entry in specs], shared_a=True)
        metas[name] = entry_meta

    def build(side, base, fields_side, constants_side, every_field):
        """One side's engine (timed wrapper), its setup time and its plan lines."""
        started = time.perf_counter()
        tf32 = args.a_tf32 if side == "a" else args.b_tf32
        ab_step_time.set_tf32(torch, False)  # built in IEEE (tf32_after_build)
        rewrites = RewriteConfig(**fields_side) if fields_side else "auto"
        integrator = (IntegratorConfig(temperature_K=args.temperature_K) if args.timestep_fs is None else
                      IntegratorConfig(temperature_K=args.temperature_K, timestep_fs=args.timestep_fs))
        planner = planners[side]
        # Every constant either side sets, at this side's value (the default where it leaves one alone), over the
        # whole build: the force field (rewrites read them), the plan (the planner's label-keyed decisions, such as
        # fastmace.planner.fixed:DECISION_66_TF32X3_LABELS, are taken in plan()) and the engine. Not only this
        # side's own constants: ConstantsSide sets them before every step and leaves them set, so a build after
        # another side's steps would otherwise read that side's values (OMAT-0 small on the H200, 8 November
        # 2026: the plan ran outside the side's constants and read whatever the last step had left, 2.2 or 5.4 ms
        # per step by build order).
        field, plan, engine = built_under(
            {**defaults, **constants_side},
            lambda: ForceField(loaded, base, precision=args.precision, rewrites=rewrites),
            lambda field: planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K),
                                       hardware),
            lambda plan, field: planner.build(plan, field, base.copy(), temperature_K=args.temperature_K),
            prepare=args.prepare)
        if tf32:
            ab_step_time.tf32_after_build(torch, engine)
        elif args.a_recapture if side == "a" else args.b_recapture:
            engine.invalidate_graphs("TF32 screen control: re-capture in IEEE")
        timed = ab_step_time.Tf32Side(ab_step_time.FastMACESide(engine, torch), tf32, torch)
        if defaults:
            timed = ConstantsSide(timed, {**defaults, **constants_side})
        lines = [line.strip() for line in plan.describe(full=True).splitlines()
                 if any(f"RewriteConfig.{f} " in line for f in every_field)]
        return timed, time.perf_counter() - started, lines

    def measure(name, sides, setup, lines, structure, natoms):
        for side in order:  # warm up outside the compile-counter window (new edge sizes compile here)
            step_time.warm_up(sides[side], protocol.warmup)

        def engine_counters():  # rebuilds and redos explain a neighbour-list A/B (fp32 profile)
            return {side: {key: getattr(sides[side].engine, key, None)
                           for key in ("rebuilds", "pipelined_redos", "pipelined_early_rebuilds")} for side in sides}

        counters_before = engine_counters()
        counted = ab_step_time.compile_counters(counters)
        result = ab.run_ab(sides, dataclasses.replace(protocol, warmup=0), pattern=args.pattern,
                           stop_after=args.stop_after, stop_beyond=args.stop_beyond)
        changes, recompiled = ab_step_time.compile_changes(counted, ab_step_time.compile_counters(counters))
        medians = {side: result.statistics[side].median_ms for side in sides}
        blocks = {side: result.statistics[side].block_ms for side in sides}
        entry = dict(natoms=natoms, ab=result.as_dict(), ratio_b_over_a=medians["b"] / medians["a"],
                     pair_ratios=[b / a for a, b in zip(blocks["a"], blocks["b"])], setup_s=setup, plan_lines=lines,
                     compile_during_timing=dict(changes=changes, recompiled=recompiled),
                     observables={side: sides[side].observables() for side in sides}, load_after=list(os.getloadavg()),
                     engine_counters_timed={side: {key: (value - counters_before[side][key]
                                                         if value is not None and counters_before[side][key] is not None
                                                         else None) for key, value in now.items()}
                                            for side, now in engine_counters().items()},
                     other_gpu_processes=[pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()])
        entry["valid"] = not recompiled and not entry["other_gpu_processes"]
        metas[name]["structures"][structure] = entry
        print(*([name] if name is not None else []), structure,
              " ".join(f"{s} {medians[s]:.4f} ms" for s in sides), f"b/a {entry['ratio_b_over_a']:.4f}",
              "valid" if entry["valid"] else "INVALID",
              f"(stopped after {result.stopped_after} blocks)" if result.stopped_after else "", flush=True)
        output = args.output if name is None else args.output.with_name(f"{args.output.name}-{name}")
        output.with_suffix(".json").write_text(json.dumps(metas[name], indent=2, default=str) + "\n")

    for spec in args.structures:
        structure, base = ab_step_time.read_structure(spec, None)
        built = {}  # side a, or the entry's name: (timed side, setup s, plan lines)
        everything = {**args.a}
        for _, fields_b, _ in specs:
            everything.update(fields_b)

        def build_b(index):
            name, fields_b, constants_b = specs[index]
            built[index] = build("b", base, fields_b, constants_b, {**args.a, **fields_b})

        # order ab: a, then each b just before its pair; order ba: every b, then a
        if order[0] == "a":
            built["a"] = build("a", base, args.a, args.a_constant, everything)
        else:
            for index in range(len(specs)):
                build_b(index)
            built["a"] = build("a", base, args.a, args.a_constant, everything)
        for index, (name, fields_b, _) in enumerate(specs):
            if index not in built:
                build_b(index)
            a_side, a_setup, a_lines = built["a"]
            b_side, b_setup, b_lines = built.pop(index)
            keep = set({**args.a, **fields_b})
            measure(name, {side: {"a": a_side, "b": b_side}[side] for side in order},
                    {"a": a_setup, "b": b_setup},
                    {"a": [line for line in a_lines if any(f"RewriteConfig.{f} " in line for f in keep)],
                     "b": b_lines}, structure, len(base))
            b_side = None
            torch.cuda.empty_cache()
        built = None
        ab_step_time.set_tf32(torch, False)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
