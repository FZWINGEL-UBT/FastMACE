"""Alternating A/B MD step time of this checkout against an earlier FastMACE release, in one process.

1b-T23 (acceptance item 5) timed 1b against the 1a release; 2-T23 (phase 2 acceptance item 8) times
this checkout against the 1b release with ``--release 1b``. Sides (--sides, two for an A/B process,
one for a control or an alternating-process run):

  --release 1a (the default):
    1b  this checkout's graph engine through the minimal planner (``--planner-1b minimal``, the
        default: what ``Simulation`` and ``fm.md`` build) or the fixed-plan planner (``fixed``, main.py's)
    1a  the release at tag ``fastmace-1a`` (bb35363), exported as the renamed package ``fastmace_1a``
        (``python -m fastmace.bench.versions export --ref fastmace-1a --destination DIR``; --renamed DIR),
        through its fixed-plan planner
  --release 1b:
    2   this checkout, through the planner ``--planner-1b`` names, as above
    1b  the release at tag ``fastmace-1b`` (673a8a1), exported as ``fastmace_1b``
        (``... export --ref fastmace-1b --name fastmace_1b ...``), through its own minimal planner, so
        both sides build what ``Simulation`` builds
  --release 2:
    3   this checkout, as above
    2   the release at tag ``fastmace-2`` (e52c482), exported as ``fastmace_2``, through its minimal planner
  --release p3b (3-T21, phase 3 item 10):
    3   this checkout, as above, with ``fixcm_projection="ase"`` (decision 37 made the mass-weighted
        projection the NVT default; the baseline has ASE's)
    p3b main before phase 3 (tag ``phase-3-baseline``, 156ab76), exported as ``fastmace_p3b``, through its
        minimal planner
  --release 3 (3b-T10, phase 3b item 5):
    3b  this checkout, as above, D3 off (no dispersion)
    3   the release at tag ``fastmace-3``, exported as ``fastmace_3``, through its minimal planner. The tag
        was not set when this was written, so its commit has no default: give --expected-commit

``--engine-option FIELD=VALUE`` (repeatable) sets an ``EngineConfig`` field on this checkout's side only
(``fused_integrator=on``: an option the release does not have); ``on``/``off`` read as booleans.
``--integrator-option FIELD=VALUE`` does the same for ``IntegratorConfig`` (release p3b pins
``fixcm_projection=ase`` unless given).

Both are FastMACE, so they patch the same process-wide objects (fastmace.bench.versions): this tool
gives each side its own patches (``ProcessPatches``) before its load, build, warm-up and every timed
block, outside the clock, and shares this process's OEq runtime with the release's copy. It records every
patch each side left, and a run is invalid when anything compiled or any patch changed while timed,
or when the two plans' engine options differ. PolarMACE (a class patch), plain cuEq (a conversion
patch per load) and NPT (the strain patch with a stress path) are refused with two sides: those run in
alternating processes, one side each (--sides 1a, then --sides 1b, ...), with the cross-process noise.
With --dump-final each side's positions and forces after its last timed block are written too: one-side
processes of the same seed and protocol compare trajectories across versions (plain cuEq float64).

Both sides start from the same structure with the same seed, at 0 K, with the thermostat at
--temperature-K. Hybrid FP32, NVT, one backend family. Use an otherwise idle GPU; no launcher is
needed. Output: <output>/metadata.json (fastmace.bench.step_time records per side, the A/B result,
ratio of this checkout over the release, plans, patches, compile counters), and with --dump-forces the
starting forces per side.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
from fastmace.bench import ab, step_time, versions

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402  (tools/: FastMACESide, compile counters, structures)


@dataclasses.dataclass(frozen=True)
class Release:
    """An earlier release as one side: tag, commit, renamed package, planner, and this checkout's side name."""

    tag: str
    commit: str
    renamed: str
    planner: str  # "fixed" or "minimal": the planner the release's own Simulation path used
    current: str  # the side name of this checkout when timed against this release


RELEASES = {
    "1a": Release("fastmace-1a", "bb35363", versions.DEFAULT_NAME, "fixed", "1b"),
    "1b": Release("fastmace-1b", "673a8a1", "fastmace_1b", "minimal", "2"),
    "2": Release("fastmace-2", "e52c482", "fastmace_2", "minimal", "3"),
    "p3b": Release("phase-3-baseline", "156ab76", "fastmace_p3b", "minimal", "3"),
    "3": Release("fastmace-3", None, "fastmace_3", "minimal", "3b"),  # commit: --expected-commit (tag not set yet)
}
# IntegratorConfig fields this checkout's side pins against a release unless --integrator-option sets them.
RELEASE_INTEGRATOR_PINS = {"p3b": {"fixcm_projection": "ase"}}
SIDES = ("1a", "1b")
FASTMACE_1A_TAG = RELEASES["1a"].tag
FASTMACE_1A_COMMIT = RELEASES["1a"].commit
RENAMED = RELEASES["1a"].renamed
ALTERNATING = "run alternating processes instead, one side each (--sides 1a, then --sides 1b, ...)"


def _is_polar(label):
    from fastmace.model.labels import LABELS

    entry = LABELS.get(label)
    return entry is not None and entry.model_type == "PolarMACE"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--release", choices=sorted(RELEASES), default="1a",
                   help="The earlier release timed against this checkout (1a: sides 1a and 1b; 1b: sides 1b and 2)")
    p.add_argument("--sides", nargs="+", choices=("1a", "1b", "2", "3", "p3b", "3b"),
                   help="Build order; two sides for an A/B process, one for a control (default: release, checkout)")
    p.add_argument("--renamed", type=Path, help="Directory holding the release's exported renamed package")
    p.add_argument("--expected-commit", help="Commit (prefix) the release's copy must be (default: the release's)")
    p.add_argument("--planner-1b", choices=("minimal", "fixed"), default="minimal")
    p.add_argument("--engine-option", action="append", default=[], metavar="FIELD=VALUE",
                   help="EngineConfig field for this checkout's side only (on/off read as booleans)")
    p.add_argument("--integrator-option", action="append", default=[], metavar="FIELD=VALUE",
                   help="IntegratorConfig field for this checkout's side only (release p3b: fixcm_projection=ase "
                        "unless given)")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"],
                   help="Benchmark names (under FASTMACE_STRUCTURES) or files with PATH@NXxNYxNZ")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12, help="Timed blocks per side (12: the 1a-T18 noise floor)")
    p.add_argument("--pattern", choices=ab.PATTERNS, default="abab")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    p.add_argument("--ensemble", choices=("nvt", "npt"), default="nvt")
    p.add_argument("--cell-mode", choices=("hydrostatic", "full"), default="hydrostatic")
    p.add_argument("--dump-forces", action="store_true", help="Write each side's starting forces per structure")
    p.add_argument("--allow-option-changes", nargs="*", default=[], metavar="OPTION",
                   help="Options whose difference between the sides is expected (rewrites a later release turned "
                        "on by default); any other difference makes the process invalid")
    p.add_argument("--dump-final", action="store_true",
                   help="Write each side's positions and forces after its last timed block, per structure")
    args = p.parse_args(argv)
    args.engine_options = {}
    for item in args.engine_option:
        field, sep, value = item.partition("=")
        if not sep or not field:
            p.error(f"--engine-option {item!r}: expected FIELD=VALUE")
        args.engine_options[field] = {"on": True, "off": False}.get(value, value)
    args.integrator_options = dict(RELEASE_INTEGRATOR_PINS.get(args.release, {}))
    for item in args.integrator_option:
        field, sep, value = item.partition("=")
        if not sep or not field:
            p.error(f"--integrator-option {item!r}: expected FIELD=VALUE")
        args.integrator_options[field] = {"on": True, "off": False}.get(value, value)
    release = RELEASES[args.release]
    args.current = release.current
    if args.sides is None:
        args.sides = [args.release, args.current]
    if args.expected_commit is None:
        args.expected_commit = release.commit
    if args.expected_commit is None and args.release in args.sides:
        p.error(f"--release {args.release}: give --expected-commit (the {release.tag} commit; no default yet)")
    allowed = (args.release, args.current)
    if (len(args.sides) not in (1, 2) or len(set(args.sides)) != len(args.sides)
            or not set(args.sides) <= set(allowed)):
        p.error(f"--sides takes one side, or the two sides {allowed[0]} and {allowed[1]} in build order "
                f"(--release {args.release})")
    if args.release in args.sides and args.renamed is None:
        p.error(f"a {args.release} side needs --renamed DIR (python -m fastmace.bench.versions export "
                f"--ref {release.tag} --name {release.renamed})")
    if len(args.sides) == 2:
        if _is_polar(args.model):
            p.error(f"PolarMACE ({args.model}) patches its class process-wide, which two versions cannot share; "
                    + ALTERNATING)
        if args.backend == "cueq":
            p.error("plain cuEq converts with a per-load patch of MACE's converter; " + ALTERNATING)
        if args.ensemble == "npt":
            p.error("NPT needs each version's strain patch while it evaluates stress; " + ALTERNATING)
    return args


def ratio_1b_over_1a(medians):
    """Median step time of 1b over that of 1a, whatever the build order."""
    return medians["1b"] / medians["1a"]


def ratio_current_over_release(medians, current, release):
    """Median step time of this checkout's side over the release's, whatever the build order."""
    return medians[current] / medians[release]


def compare_options(first, second, allowed=()):
    """Equality of two plans' engine options ({option: value}), with each differing option's two values.

    ``allowed`` names options whose difference is expected (rewrites that a later release turned on by
    default, e.g. ``gated_rows``; an option counts with its ``off:`` key too): ``unexpected`` lists the
    other differences, and ``as_expected`` is true when there are none."""
    differences = {key: [first.get(key), second.get(key)] for key in sorted(set(first) | set(second))
                   if first.get(key) != second.get(key)}
    unexpected = sorted(key for key in differences if key.removeprefix("off:") not in set(allowed))
    return {"equal": not differences, "differences": differences, "allowed": sorted(allowed),
            "unexpected": unexpected, "as_expected": not unexpected}


class Version:
    """One side's planner, loaded model and engine builder."""

    def __init__(self, side, args, package):
        self.side, self.args, self.package = side, args, package
        self.loaded = self.planner = self.hardware = self.force_field_class = None
        self.cache = None
        if side == args.current:
            self.minimal = args.planner_1b == "minimal"
        else:
            self.minimal = RELEASES[args.release].planner == "minimal"

    def module(self, name):
        """This side's ``<package>.<name>`` (fastmace.config or fastmace_1a.config, ...)."""
        import importlib
        return importlib.import_module(f"{self.package.__name__}.{name}")

    def integrator_options(self):
        """The IntegratorConfig fields this side sets beyond the protocol's (this checkout's side only)."""
        return dict(getattr(self.args, "integrator_options", {})) if self.side == self.args.current else {}

    def prepare(self):
        """The planner and its allocator rule, before the first CUDA allocation."""
        config = self.module("config")
        options = dict(getattr(self.args, "engine_options", {})) if self.side == self.args.current else {}
        if getattr(self.args, "device", None) == "cpu":  # tools/trajectory_versions.py --device cpu, both sides:
            options.update(device="cpu", compile_mode="none")  # the CPU tests' planner (no Inductor on the CPU)
        engine = config.EngineConfig(backend=self.args.backend, **options)
        if self.minimal:
            self.planner = self.module("planner.minimal").MinimalPlanner(engine=engine)
        else:
            self.planner = self.module("planner.fixed").FixedPlanner(engine=engine)
        if getattr(self.args, "reproducible", False):
            # A reproducible plan needs its process prepared before CUDA starts (SPEC §8.7), and hardware() starts
            # it: claim the process first, as the planner's build would (3b-T10, tools/trajectory_versions.py).
            self.module("planner.reproducible").prepare_process(self.planner.process,
                                                                 owner="a reproducible MD plan (trajectory_versions)")
        self.hardware = self.planner.hardware()

    def load(self, checkpoint):
        started = time.perf_counter()
        if self.minimal:
            loading = self.module("model.loading")
            self.loaded = loading.load_pristine(self.args.model, checkpoint=str(checkpoint), sha256=_sha256(checkpoint))
        else:
            self.loaded = self.planner.load(self.args.model, checkpoint=str(checkpoint))
            self.cache = self.module("model.compile").CompiledModelCache()
        return dict(load_s=time.perf_counter() - started, sha256=self.loaded.sha256,
                    package=self.package.__name__, planner=type(self.planner).__name__)

    def build(self, base):
        config = self.module("config")
        integrator = config.IntegratorConfig(ensemble=self.args.ensemble, cell_mode=self.args.cell_mode,
                                             seed=self.args.seed, **self.integrator_options())
        fixed = self.module("planner.fixed")
        if self.minimal:
            ForceField = self.module("model.forcefield").ForceField
            Workload = self.module("planner.workload").Workload
            force_field = ForceField(self.loaded, base, precision="float32" if self.args.backend == "hybrid"
                                     else "float64")
            integrator = dataclasses.replace(integrator, temperature_K=self.args.temperature_K)
            # A ramped run (tools/trajectory_versions.py) plans for its highest target.
            highest = getattr(self.args, "max_temperature_K", None) or self.args.temperature_K
            extra = {"reproducible": True} if getattr(self.args, "reproducible", False) else {}
            plan = self.planner.plan(force_field, base, integrator, Workload(max_temperature_K=highest, **extra),
                                     self.hardware)
            engine = self.planner.build(plan, force_field, base.copy(), temperature_K=self.args.temperature_K)
        else:
            plan = self.planner.plan(self.loaded, base, integrator, self.hardware)
            engine = self.planner.build(plan, self.loaded, base.copy(), temperature_K=self.args.temperature_K,
                                        compile_cache=self.cache)
        options = {name: True for name in fixed.enabled_options(plan)}
        options.update({f"off:{name}": reason for name, reason in fixed.auto_options_off(plan).items()})
        return engine, dict(describe=plan.describe(), options=options)


def main(argv=None):
    args = parse_args(argv)
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    hybrid = args.backend == "hybrid"
    from fastmace import _patches
    from fastmace.model import loading
    oeq_runtime = str(loading.prepare_hybrid()) if hybrid else None  # 1b's, before anything imports MACE
    release = RELEASES[args.release]
    packages, source, shared = {args.current: fastmace}, None, None
    if args.release in args.sides:
        old, source = versions.load_renamed(args.renamed, release.renamed, args.expected_commit)
        import importlib
        modules = ["config", "planner.fixed", "model.compile", "_patches"]
        if release.planner == "minimal":
            modules += ["planner.minimal", "model.loading", "model.forcefield", "planner.workload"]
        for name in modules:
            importlib.import_module(f"{release.renamed}.{name}")
        packages[args.release] = old
        if hybrid:
            shared = versions.share_record(_patches, old._patches, "oeq_runtime")
    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    patches = versions.ProcessPatches()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    names = list(args.sides)
    meta = dict(schema_version=step_time.SCHEMA_VERSION, status="running", tool="ab_fastmace",
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename, python=sys.executable,
                fastmace_commit=ab_step_time.fastmace_commit(), release=dataclasses.asdict(release),
                release_source=source, current_side=args.current, release_side=args.release,
                **({"fastmace_1a": source} if args.release == "1a" else {}), model=args.model,
                checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint), backend=args.backend,
                dtype="float32" if hybrid else "float64", tf32=False, ensemble=args.ensemble,
                temperature_K=args.temperature_K, protocol=protocol.as_dict(), pattern=args.pattern, sides=names,
                build_order=names, planner_1b=args.planner_1b, oeq_runtime=oeq_runtime, oeq_runtime_shared=shared,
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                versions={}, structures={})
    path = args.output / "metadata.json"

    def save():
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    try:
        sides_by_name = {name: Version(name, args, packages[name]) for name in names}
        for name in names:
            patches.enter(name)
            sides_by_name[name].prepare()
            patches.leave(name)
        import torch
        from torch._dynamo.utils import counters
        for name in names:
            patches.enter(name)
            meta["versions"][name] = sides_by_name[name].load(checkpoint)
            patches.leave(name)
        meta.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        save()
        for spec in args.structures:
            structure, base = ab_step_time.read_structure(spec, None)
            entry = meta["structures"][structure] = dict(natoms=len(base), structure=str(spec), sides={}, plans={})
            torch.cuda.reset_peak_memory_stats()
            sides, setup, warmup_s = {}, {}, {}
            for name in names:
                patches.enter(name)
                started = time.perf_counter()
                engine, entry["plans"][name] = sides_by_name[name].build(base)
                sides[name] = ab_step_time.FastMACESide(engine, torch)
                setup[name] = time.perf_counter() - started
                if args.dump_forces:
                    directory = args.output / "forces"
                    directory.mkdir(exist_ok=True)
                    np.save(directory / f"{structure}-{name}.npy", engine.state.forces.detach().cpu().numpy())
                started = time.perf_counter()
                step_time.warm_up(sides[name], protocol.warmup)
                warmup_s[name] = time.perf_counter() - started
                patches.leave(name)
            if len(names) == 2:
                entry["plan_check"] = compare_options(entry["plans"][args.release]["options"],
                                                      entry["plans"][args.current]["options"],
                                                      allowed=args.allow_option_changes)
                entry["plan_describe_equal"] = (entry["plans"][args.release]["describe"]
                                                == entry["plans"][args.current]["describe"])
            changed = []

            def before(name):
                if patches.current is not None:
                    changed.extend(f"{patches.current}:{key}" for key in patches.leave(patches.current))
                patches.enter(name)

            timed = dataclasses.replace(protocol, warmup=0)
            counted = ab_step_time.compile_counters(counters)
            started = time.perf_counter()
            if len(names) == 2:
                result = ab.run_ab(sides, timed, pattern=args.pattern, before=before)
                entry["ab"] = result.as_dict()
                statistics_by_side = result.statistics
                medians = {n: s.median_ms for n, s in statistics_by_side.items()}
                entry["ratio"] = ratio_current_over_release(medians, args.current, args.release)
                entry[f"ratio_{args.current}_over_{args.release}"] = entry["ratio"]
            else:
                before(names[0])
                statistics_by_side = {names[0]: step_time.time_engine(sides[names[0]], timed)}
            before(names[0])  # record what the last side left
            entry["timing_wall_s"] = time.perf_counter() - started
            entry["warmup_s"] = warmup_s
            changes, recompiled = ab_step_time.compile_changes(counted, ab_step_time.compile_counters(counters))
            entry["compile_during_timing"] = dict(changes=changes, recompiled=recompiled)
            entry["patches_changed_while_timed"] = changed
            entry["valid"] = (not recompiled and not changed
                              and entry.get("plan_check", {"as_expected": True})["as_expected"])
            peak = torch.cuda.max_memory_reserved()
            for name in names:
                info = dict(sides[name].engine.info())
                info["force_field"] = dict(sides[name].engine.force_field.info())
                entry["sides"][name] = step_time.result_record(
                    statistics_by_side[name], setup_s=setup[name], peak_memory_bytes=peak,
                    observables=sides[name].observables(), info=info)
            if args.dump_final:
                directory = args.output / "final"
                directory.mkdir(exist_ok=True)
                for name in names:
                    sides[name].engine.flush()
                    state = sides[name].engine.state
                    np.save(directory / f"{structure}-{name}-positions.npy", state.positions.detach().cpu().numpy())
                    np.save(directory / f"{structure}-{name}-forces.npy", state.forces.detach().cpu().numpy())
            if len(names) == 2:
                a, b = (entry["sides"][name]["observables"] for name in names)
                entry["observables_difference"] = {key: abs(a[key] - b[key]) for key in a}
            entry["patches"] = patches.as_dict()
            entry["other_gpu_processes"] = [pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()]
            save()
            print(structure, " ".join(f"{name} {entry['sides'][name]['median_ms']:.3f} ms" for name in names),
                  f"{args.current}/{args.release} {entry['ratio']:.4f}" if "ratio" in entry else "",
                  "valid" if entry["valid"] else "INVALID", flush=True)
            sides = engine = None
            torch.cuda.empty_cache()
        meta["patches_applied"] = {name: [str(record) for record in sides_by_name[name].module("_patches").applied()]
                                   for name in names}
        meta["status"] = "complete"
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return meta


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
