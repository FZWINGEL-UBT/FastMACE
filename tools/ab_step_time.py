"""Alternating A/B MD step time in one process (SPEC §12.5): the research engine against FastMACE.

Both sides are built and warmed up, then their timed blocks alternate (fastmace.bench.ab.run_ab).
Each side loads its own calculator, so the two never share a compiled model, as two different
engines would not. Sides (--sides, two, or one for a single-engine control run):

  baseline  the research engine's GPULangevin (main.py's gpu default for the model, NVT) from an
            exported baseline (--baseline DIR, with a COMMIT file)
  fastmace  FastMACE's engine; refused until it exists (plan task 1a-T13)

Until T13, baseline against baseline measures the noise floor of the 3 % criterion. Both sides then
start from the same structure with the same seed, so they do the same work; their final temperature
and potential energy are recorded to show that neither disturbed the other (shared compiled modules,
the streaming counter of bind_streaming_count).

Run it with the hybrid environment through the export's studies/hybrid_oeq/scripts/oeq_no_nvcc.sh and
PYTHONPATH=<FastMACE checkout>, on an otherwise idle GPU. Output: <output>/metadata.json.

The timing loop is the research engine's studies/other_models/probes/model_step_time.py, refactored
into fastmace.bench.step_time and fastmace.bench.ab.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
from fastmace.bench import ab, step_time

FLAGS = ("factor_species", "fold_cutoff", "sort_receivers", "density_first", "reassociate_first",
         "joint_radial_input", "fused_folded_backward", "segmented_first_sum", "gpu_neighbour_list",
         "block_linear", "grouped_linear", "merged_residual", "streaming_conv", "fused_radial_hidden", "pipelined")


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


def gpu_processes():
    """PIDs of the compute processes on the GPU."""
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    return [int(line) for line in output.split() if line.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--baseline", required=True, type=Path,
                   help="Exported research-engine directory with a COMMIT file")
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
    p.add_argument("--main-args", default="", help="Further main.py options for the baseline sides")
    args = p.parse_args(argv)
    if len(args.sides) not in (1, 2):
        p.error("--sides takes one side (control run) or two (A/B)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if "fastmace" in args.sides:
        raise SystemExit("--sides fastmace: FastMACE has no engine yet (plan task 1a-T13)")
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    export = ab.load_baseline(args.baseline)
    ab.activate(export)
    import langevin
    ab.check_origin(export, "langevin", langevin.__file__)
    from langevin import compile_cache
    from langevin import workflow as experiment

    if gpu_processes():
        raise RuntimeError("GPU must be idle")
    experiment.require_hybrid_wrapper()
    args.output.mkdir(parents=True, exist_ok=False)
    cache_info = compile_cache.configure(args.output)
    from langevin import gpu_langevin
    cache_info["cuda_allocator"] = gpu_langevin.configure_allocator()  # before the first CUDA allocation
    import torch
    from torch._dynamo.utils import counters

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    md = experiment.parse_args(["--model", args.model, "--checkpoint", str(checkpoint)] + args.main_args.split())
    experiment.apply_polar_options(md)
    names = [f"{side}_{index}" if args.sides.count(side) > 1 else side for index, side in enumerate(args.sides)]
    meta = dict(schema_version=step_time.SCHEMA_VERSION, status="running",
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                python=sys.executable, fastmace_commit=fastmace_commit(), baseline=export.as_dict(),
                model=args.model, checkpoint=str(checkpoint), checkpoint_sha256=experiment.sha256_file(checkpoint),
                backend="hybrid", dtype="float32", tf32=False, ensemble="nvt", temperature_K=args.temperature_K,
                protocol=protocol.as_dict(), pattern=args.pattern, sides=names,
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                compile_cache=cache_info, calculators={}, structures={})
    path = args.output / "metadata.json"

    def save():
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    try:
        calcs, flags, common = {}, {}, None
        for name in names:
            started = time.perf_counter()
            if common is None:
                common, calc = experiment.load_md_calculator("float32", True, "cuda:0", checkpoint,
                                                             label=args.model)
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
        torch.set_default_dtype(torch.float32)
        meta.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        save()
        for spec in args.structures:
            structure, base = experiment.read_structure_spec(spec)
            entry = meta["structures"][structure] = dict(natoms=len(base), structure=str(spec), sides={})
            torch.cuda.reset_peak_memory_stats()
            sides, setup = {}, {}
            for name in names:
                started = time.perf_counter()
                engine = experiment.GPUEngine(base.copy(), calcs[name], timestep_fs=md.timestep_fs,
                                              friction=md.friction, seed=args.seed, skin=md.skin,
                                              compile_mode=md.compile_mode, device="cuda:0", **flags[name])
                sides[name] = ResearchEngineSide(engine, args.temperature_K, torch)
                setup[name] = time.perf_counter() - started
            started = time.perf_counter()
            if len(names) == 2:
                result = ab.run_ab(sides, protocol, pattern=args.pattern)
                entry["ab"] = result.as_dict()
                statistics_by_side = result.statistics
            else:
                statistics_by_side = {names[0]: step_time.time_engine(sides[names[0]], protocol)}
            entry["timing_wall_s"] = time.perf_counter() - started
            # Warm-up includes compilation and capture; the timed blocks start after it.
            peak = torch.cuda.max_memory_reserved()
            for name in names:
                entry["sides"][name] = step_time.result_record(
                    statistics_by_side[name], setup_s=setup[name], peak_memory_bytes=peak,
                    observables=sides[name].observables(), info=sides[name].engine.info())
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
            sides = None
            torch.cuda.empty_cache()
        meta["status"] = "complete"
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["compile_cache"]["stats"] = compile_cache.cache_stats()
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
