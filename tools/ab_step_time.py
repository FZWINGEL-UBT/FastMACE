"""Alternating A/B MD step time in one process (SPEC §12.5): the research engine against FastMACE.

Both sides are built and warmed up, then their timed blocks alternate (fastmace.bench.ab.run_ab).
Each side loads its own calculator, so the two never share a compiled model, as two different
engines would not. Sides (--sides, two, or one for a single-engine control run):

  baseline  the research engine's GPULangevin (main.py's gpu default for the model) from an
            exported baseline (--baseline DIR, with a COMMIT file)
  fastmace  FastMACE's GraphEngine through the fixed-plan planner (plan task 1a-T15), main.py's
            plan for the model

The sides are built in the order given, so "--sides fastmace baseline" swaps the build order (1a-T18
saw a 0.1-0.16 % offset tied to it). Both start from the same structure with the same seed, so they
do the same work; their final temperature and potential energy are recorded to show that neither
disturbed the other (shared compiled modules, the streaming counter of bind_streaming_count, the
process-wide MACE patches of both engines). --dump-forces writes FastMACE's starting forces per
structure (forces/<structure>-<side>.npy), to compare an A/B process with a FastMACE-only one.
--ensemble npt runs both sides with the Langevin-Hoover BAOAB barostat (--cell-mode).

FastMACE prepares its OEq runtime in-process before anything imports MACE, so a process with a
FastMACE side does that first, whatever the build order. Run an A/B process with the hybrid
environment through the export's studies/hybrid_oeq/scripts/oeq_no_nvcc.sh (the research engine
needs it) and PYTHONPATH=<FastMACE checkout>; a FastMACE-only process (--sides fastmace) needs
neither the launcher nor --baseline. Use an otherwise idle GPU. Output: <output>/metadata.json.

The timing loop is the research engine's studies/other_models/probes/model_step_time.py, refactored
into fastmace.bench.step_time and fastmace.bench.ab.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
from fastmace.bench import ab, step_time

# Every engine option main.py passes to GPUEngine (workflow.run_structure at the baseline tag), so the
# baseline side runs main.py's plan. species_skip was missing until 1a-T19: it is on by default for the
# gpu engine where the model supports it (MP-0, POLAR-1, OMOL-0, MACE-OFF; not MH-1).
FLAGS = ("factor_species", "fold_cutoff", "sort_receivers", "density_first", "reassociate_first",
         "joint_radial_input", "fused_folded_backward", "segmented_first_sum", "gpu_neighbour_list",
         "block_linear", "grouped_linear", "merged_residual", "species_skip", "streaming_conv",
         "fused_radial_hidden", "speculative_graph", "pipelined")
NPT_KEYWORDS = ("pressure_GPa", "barostat_tau_fs", "barostat_mass_factor")


class ResearchEngineSide:
    """A research-engine GPUEngine as a fastmace.bench.step_time.TimedEngine."""

    def __init__(self, engine, temperature_K, torch):
        self.engine, self.temperature_K, self.torch = engine, temperature_K, torch

    def step(self):
        self.engine.step(self.temperature_K)

    def prepare_block(self):
        # Fields built from one compiled model share its streaming modules; the field stepped next
        # must own their edge counter (gpu_langevin.MACEForces.bind_streaming_count).
        self.engine.field.bind_streaming_count()

    def finish_block(self):
        self.engine.dyn.flush_speculative()

    def synchronize(self):
        self.torch.cuda.synchronize()

    def observables(self):
        temperature, potential, kinetic = self.engine.observables()[:3]
        return dict(temperature_K=float(temperature), potential_eV_atom=float(potential),
                    kinetic_eV_atom=float(kinetic))


class FastMACESide:
    """A FastMACE GraphEngine as a fastmace.bench.step_time.TimedEngine."""

    def __init__(self, engine, torch):
        self.engine, self.torch = engine, torch

    def step(self):
        self.engine.step()

    def prepare_block(self):
        pass  # the force field owns its edge counter (per_engine_edge_counter)

    def finish_block(self):
        self.engine.flush()

    def synchronize(self):
        self.torch.cuda.synchronize()

    def observables(self):
        # As the research engine's GPUEngine.observables: kinetic energy from the velocities, 3N degrees of freedom.
        from ase import units
        self.engine.flush()
        state = self.engine.state
        n = state.positions.shape[0]
        kinetic = 0.5 * float((state.masses.reshape(-1, 1) * state.velocities ** 2).sum())
        return dict(temperature_K=2 * kinetic / (3 * n * units.kB), potential_eV_atom=float(state.energy[0]) / n,
                    kinetic_eV_atom=kinetic / n)


class CustomPassSlot:
    """Gives each side its own key in Inductor's single ``post_grad_custom_pre_pass`` slot (1a-T19, option (a)).

    Both engines keep Inductor's ``mm + full-matrix add`` unfused on the GPU and put a no-op custom
    pass into that one config slot, whose uuid keys the compiler cache: the research engine
    "langevin-unfused-full-addmm-v2", FastMACE "fastmace-unfused-full-addmm-v1". Each refuses a slot
    that holds the other's key, so the two cannot share a process unchanged. The A/B tool therefore
    puts a side's own key (none before its first build) into the slot before that side loads, builds
    and warms up (where it compiles), and keeps what the side left there. The key matters only at
    compile time, so the timed blocks must not compile: ``compile_changes`` checks it.
    """

    def __init__(self, inductor_config=None):
        if inductor_config is None:
            from torch._inductor import config as inductor_config
        self.config = inductor_config
        self.keys = {}

    def enter(self, name):
        self.config.post_grad_custom_pre_pass = self.keys.get(name)

    def leave(self, name):
        self.keys[name] = self.config.post_grad_custom_pre_pass

    def uuids(self):
        return {name: key.uuid() if hasattr(key, "uuid") else None for name, key in self.keys.items()}


COMPILE_COUNTER_GROUPS = ("stats", "recompiles", "frames", "inductor", "aot_autograd", "graph_break")
# Counters that move only when something is compiled; any change during the timed blocks invalidates a run.
COMPILE_SIGNALS = (("stats", "unique_graphs"), ("frames", "ok"), ("frames", "total"))


def compile_counters(counters):
    """A copy of dynamo's compile counters (torch._dynamo.utils.counters)."""
    return {group: dict(counters[group]) for group in COMPILE_COUNTER_GROUPS if group in counters}


def compile_changes(before, after):
    """(changes, recompiled): every counter that moved between two snapshots, and whether one of them means a
    compilation (a new graph, a compiled frame, or any recompile counter)."""
    changes = {}
    for group in sorted(set(before) | set(after)):
        old, new = before.get(group, {}), after.get(group, {})
        for key in sorted(set(old) | set(new)):
            if old.get(key, 0) != new.get(key, 0):
                changes[f"{group}.{key}"] = [old.get(key, 0), new.get(key, 0)]
    recompiled = any(f"{group}.{key}" in changes for group, key in COMPILE_SIGNALS) or any(
        name.startswith("recompiles.") for name in changes)
    return changes, recompiled


def load_another_calculator(common, label):
    """A second hybrid FP32 calculator, as workflow.load_md_calculator builds it, without its thread setup."""
    from langevin import upstream_mace
    calc = common.load_calculator(label, dtype="float32", enable_cueq=True, enable_oeq=True)
    if not any(type(module).__module__.startswith("openequivariance") for module in calc.models[0].modules()):
        raise RuntimeError("Hybrid conversion produced no OpenEquivariance modules")
    upstream_mace.require_fused_cueq(calc.models[0])
    return calc


def fastmace_commit():
    """The FastMACE commit of this checkout: git, else the NAS copy's .source_commit."""
    root = Path(fastmace.__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = root / ".source_commit"
        return marker.read_text().strip() if marker.is_file() else None


def in_container(environ=os.environ, dockerenv="/.dockerenv"):
    """True inside a Modal container (MODAL_IS_REMOTE, MODAL_TASK_ID) or a Docker one (/.dockerenv)."""
    return bool(environ.get("MODAL_IS_REMOTE") or environ.get("MODAL_TASK_ID")) or os.path.exists(dockerenv)


def gpu_processes(container=None):
    """PIDs of the compute processes on the GPU.

    Inside a container, nvidia-smi lists every GPU process as PID 1 (the container's init; Modal L4 check,
    results/modal-opt-T08-L4-3e5decd/pid-check.txt), our own included, so PID 1 is dropped there: a container's
    init never runs on the GPU. On a workstation the list is nvidia-smi's, unchanged."""
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    pids = [int(line) for line in output.split() if line.strip()]
    if in_container() if container is None else container:
        pids = [pid for pid in pids if pid != 1]
    return pids


def prepare_fastmace_hybrid():
    """FastMACE's in-process OEq runtime, before anything imports MACE (fastmace.model.loading.prepare_hybrid)."""
    from fastmace.model import loading
    return str(loading.prepare_hybrid())


def fastmace_planner(args):
    """The fixed-plan planner of main.py's defaults for --backend: dtype and every option on auto."""
    from fastmace import config
    from fastmace.planner.fixed import FixedPlanner
    return FixedPlanner(engine=config.EngineConfig(backend=args.backend))


def fastmace_integrator(args):
    from fastmace import config
    return config.IntegratorConfig(ensemble=args.ensemble, cell_mode=args.cell_mode, seed=args.seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--baseline", type=Path,
                   help="Exported research-engine directory with a COMMIT file (needed for a baseline side)")
    p.add_argument("--sides", nargs="+", choices=("baseline", "fastmace"), default=["baseline", "fastmace"])
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--structures", nargs="+", default=["NaCl", "NaNbO3", "NbCl5"],
                   help="Benchmark names (3 x 3 x 3), or structure files as PATH@NXxNYxNZ")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=4, help="Timed blocks per side")
    p.add_argument("--pattern", choices=ab.PATTERNS, default="abab")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid",
                   help="hybrid: main.py's gpu default (FP32); cueq: plain cuEquivariance float64 (no launcher)")
    p.add_argument("--ensemble", choices=("nvt", "npt"), default="nvt")
    p.add_argument("--cell-mode", choices=("hydrostatic", "full"), default="hydrostatic", help="NPT cell mode")
    p.add_argument("--dump-forces", action="store_true",
                   help="Write each FastMACE side's starting forces to forces/<structure>-<side>.npy")
    p.add_argument("--main-args", default="", help="Further main.py options for the baseline sides")
    args = p.parse_args(argv)
    if len(args.sides) not in (1, 2):
        p.error("--sides takes one side (control run) or two (A/B)")
    if "baseline" in args.sides and args.baseline is None:
        p.error("a baseline side needs --baseline DIR")
    return args


def launched_by_hybrid_wrapper(environ=os.environ):
    """Whether this process was started through oeq_no_nvcc.sh: the variables it exports, read before FastMACE's
    in-process OEq runtime sets some of the same names (call this first)."""
    return all(environ.get(name) for name in ("OEQ_NVCC_ATTEMPTS", "CUDA_HOME", "CUDACXX"))


def main(argv=None):
    launched = launched_by_hybrid_wrapper()  # before prepare_fastmace_hybrid changes the environment
    args = parse_args(argv)
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    has_baseline, has_fastmace = "baseline" in args.sides, "fastmace" in args.sides
    hybrid = args.backend == "hybrid"
    oeq_runtime = prepare_fastmace_hybrid() if has_fastmace and hybrid else None  # before anything imports MACE
    export = experiment = compile_cache = None
    if has_baseline:
        export = ab.load_baseline(args.baseline)
        ab.activate(export)
        import langevin
        ab.check_origin(export, "langevin", langevin.__file__)
        from langevin import compile_cache
        from langevin import workflow as experiment

    if gpu_processes():
        raise RuntimeError("GPU must be idle")
    if has_baseline and hybrid:
        experiment.require_hybrid_wrapper()
    args.output.mkdir(parents=True, exist_ok=False)
    cache_info = {}
    if has_baseline:
        cache_info = compile_cache.configure(args.output)
        from langevin import gpu_langevin
        cache_info["cuda_allocator"] = gpu_langevin.configure_allocator()  # before the first CUDA allocation
    planner = fastmace_planner(args) if has_fastmace else None
    hardware = planner.hardware() if has_fastmace else None  # FastMACE's allocator rule, then CUDA
    import torch
    from torch._dynamo.utils import counters

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    md = None
    if has_baseline:
        md = experiment.parse_args(["--model", args.model, "--checkpoint", str(checkpoint), "--backend", args.backend,
                                    "--ensemble", args.ensemble, "--cell-mode", args.cell_mode]
                                   + args.main_args.split())
        experiment.apply_polar_options(md)
        if args.ensemble == "npt":
            from langevin import upstream_mace
            # main.py's stress path, before any model is built (process-wide).
            upstream_mace.use_single_graph_strain(md.single_graph_strain)
    names = [f"{side}_{index}" if args.sides.count(side) > 1 else side for index, side in enumerate(args.sides)]
    kinds = dict(zip(names, args.sides))
    meta = dict(schema_version=step_time.SCHEMA_VERSION, status="running",
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                python=sys.executable, fastmace_commit=fastmace_commit(),
                baseline=export.as_dict() if export else None, model=args.model, checkpoint=str(checkpoint),
                checkpoint_sha256=_sha256(checkpoint), backend=args.backend,
                dtype="float32" if hybrid else "float64", tf32=False,
                ensemble=args.ensemble, cell_mode=args.cell_mode if args.ensemble == "npt" else None,
                temperature_K=args.temperature_K, protocol=protocol.as_dict(), pattern=args.pattern, sides=names,
                side_kinds=kinds, build_order=names, oeq_runtime=oeq_runtime,
                hybrid_launcher=launched,
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                compile_cache=cache_info, calculators={}, structures={})
    path = args.output / "metadata.json"

    def save():
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    try:
        calcs, flags, common, loaded, cache = {}, {}, None, {}, None
        slot = CustomPassSlot()
        for name in names:
            slot.enter(name)
            started = time.perf_counter()
            if kinds[name] == "fastmace":
                from fastmace.model.compile import CompiledModelCache
                loaded[name] = planner.load(args.model, checkpoint=str(checkpoint))
                cache = CompiledModelCache()
                meta["calculators"][name] = dict(load_s=time.perf_counter() - started, backend=loaded[name].backend,
                                                 dtype=loaded[name].dtype, sha256=loaded[name].sha256)
                slot.leave(name)
                continue
            if common is None:
                common, calc = experiment.load_md_calculator(md.dtype, hybrid, "cuda:0", checkpoint, label=args.model,
                                                             cueq_conv_fusion=None if hybrid else md.cueq_conv_fusion)
            elif not hybrid:
                raise SystemExit("two baseline sides are implemented for the hybrid only")
            else:
                # load_md_calculator sets the process-wide thread settings, which works once per
                # process; later loads go to common directly, with its checks.
                calc = load_another_calculator(common, args.model)
            experiment.configure_md_calculator(calc, compute_stress=False)
            options = copy.copy(md)
            off = experiment.adapt_options_to_model(options, calc.models[0])
            calcs[name], flags[name] = calc, {flag: getattr(options, flag) for flag in FLAGS}
            meta["calculators"][name] = dict(load_s=time.perf_counter() - started, auto_options_off=off,
                                             flags=flags[name])
            slot.leave(name)
        if has_baseline and hybrid:
            torch.set_default_dtype(torch.float32)  # as before: the research engine's float32 calculator
        meta.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        save()
        for spec in args.structures:
            structure, base = read_structure(spec, experiment)
            entry = meta["structures"][structure] = dict(natoms=len(base), structure=str(spec), sides={}, plans={})
            torch.cuda.reset_peak_memory_stats()
            sides, setup, warmup_s = {}, {}, {}
            for name in names:
                slot.enter(name)
                started = time.perf_counter()
                if kinds[name] == "fastmace":
                    plan = planner.plan(loaded[name], base, fastmace_integrator(args), hardware)
                    engine = planner.build(plan, loaded[name], base.copy(), temperature_K=args.temperature_K,
                                           compile_cache=cache)
                    sides[name] = FastMACESide(engine, torch)
                    entry["plans"][name] = plan.describe()
                else:
                    engine = experiment.GPUEngine(base.copy(), calcs[name], timestep_fs=md.timestep_fs,
                                                  friction=md.friction, seed=args.seed, skin=md.skin,
                                                  compile_mode=md.compile_mode, device="cuda:0",
                                                  ensemble=args.ensemble, cell_mode=args.cell_mode,
                                                  **{key: getattr(md, key) for key in NPT_KEYWORDS}, **flags[name])
                    sides[name] = ResearchEngineSide(engine, args.temperature_K, torch)
                setup[name] = time.perf_counter() - started
                if args.dump_forces and kinds[name] == "fastmace":
                    directory = args.output / "forces"
                    directory.mkdir(exist_ok=True)
                    np.save(directory / f"{structure}-{name}.npy", engine.state.forces.detach().cpu().numpy())
                # Each side warms up (compiles and captures) right after its build, with its own key in
                # the slot; the timed blocks then run with no warm-up of their own.
                started = time.perf_counter()
                step_time.warm_up(sides[name], protocol.warmup)
                warmup_s[name] = time.perf_counter() - started
                slot.leave(name)
            entry["custom_pass_uuids"] = slot.uuids()
            timed = dataclasses.replace(protocol, warmup=0)
            before = compile_counters(counters)
            started = time.perf_counter()
            if len(names) == 2:
                result = ab.run_ab(sides, timed, pattern=args.pattern)
                entry["ab"] = result.as_dict()
                statistics_by_side = result.statistics
            else:
                statistics_by_side = {names[0]: step_time.time_engine(sides[names[0]], timed)}
            entry["timing_wall_s"] = time.perf_counter() - started
            entry["warmup_s"] = warmup_s
            changes, recompiled = compile_changes(before, compile_counters(counters))
            entry["compile_during_timing"] = dict(changes=changes, recompiled=recompiled, valid=not recompiled)
            # Warm-up includes compilation and capture; the timed blocks start after it.
            peak = torch.cuda.max_memory_reserved()
            for name in names:
                info = dict(sides[name].engine.info())
                if kinds[name] == "fastmace":
                    info["force_field"] = dict(sides[name].engine.force_field.info())
                entry["sides"][name] = step_time.result_record(
                    statistics_by_side[name], setup_s=setup[name], peak_memory_bytes=peak,
                    observables=sides[name].observables(), info=info)
            if len(names) == 2:
                a, b = (entry["sides"][name]["observables"] for name in names)
                entry["observables_difference"] = {key: abs(a[key] - b[key]) for key in a}
            entry["dynamo_counters"] = {group: dict(counters[group]) for group in ("stats", "recompiles")
                                        if group in counters}
            entry["other_gpu_processes"] = [pid for pid in gpu_processes() if pid != os.getpid()]
            save()
            print(structure, " ".join(f"{name} {entry['sides'][name]['median_ms']:.3f} ms "
                                      f"(spread {entry['sides'][name]['spread']:.2%})" for name in names),
                  f"ratio {entry['ab']['ratio_b_over_a']:.4f}" if "ab" in entry else "", flush=True)
            sides = engine = None
            torch.cuda.empty_cache()
        if has_fastmace:
            from fastmace import _patches
            meta["fastmace_patches"] = [str(record) for record in _patches.applied()]
        meta["status"] = "complete"
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if compile_cache is not None:
            meta["compile_cache"]["stats"] = compile_cache.cache_stats()
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


def read_structure(spec, experiment):
    """(name, periodic supercell) of a structure argument: the research engine's reader, or, in a FastMACE-only
    process, the same rule from tools/force_stress_gate.py, with benchmark names under FASTMACE_STRUCTURES (a
    directory holding Structures/Benchmark/R2SCAN, such as the baseline export)."""
    if experiment is not None:
        return experiment.read_structure_spec(spec)
    import importlib.util

    from ase.io import read
    location = Path(__file__).with_name("force_stress_gate.py")
    module_spec = importlib.util.spec_from_file_location("_force_stress_gate", location)
    gate = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(gate)
    name, source, repeat = gate.structure_spec(spec, (3, 3, 3), os.environ.get("FASTMACE_STRUCTURES", "."))
    atoms = read(source).repeat(repeat)
    atoms.set_pbc(True)
    return name, atoms


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
