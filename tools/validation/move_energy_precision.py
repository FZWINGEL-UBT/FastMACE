"""float32 MC energy differences against float64 (plan task 3-T11; SPEC §16 row 3 acceptance item 3, §10.4
"Energy precision", §12.3 "move-energy differences").

For one model and each structure: the engine is planned and built as ``Simulation`` builds it for an MC
workload over the superset (default {Na, Nb, O, Cl}, conductor answer 3-T11 7), runs NVT MD at each
temperature in turn (default 300 K, then 1500 K) and, after each, measures the move-energy gate's sample
moves (``planner.selfcheck.sample_moves``: every superset pair swapped, each present species transmuted to
every other superset species, displacements of ``--max-displacement-A``) on that snapshot:

- ΔE_ref: the float64 reference (``selfcheck.move_references``: the gate's ladder and child process);
- ΔE_prod: the engine's MC evaluator in float32 (its own force field, captured energy-only graph) and in
  float64 (the float64 copy of the same plan, ``fastmace.mc.evaluator``);
- per move type the worst |ΔE_prod - ΔE_ref|, in eV and in k_B T_min (default T_min 250 K, the annealing
  protocol's), and whether the gate's 0.01 k_B T_min holds.

    PYTHONPATH=$PWD FASTMACE_STRUCTURES=~/fastmace-baseline/1a-T18 python tools/validation/move_energy_precision.py \\
        --model mh1 --head matpes_r2scan --checkpoint ~/models/mace-mh-1.model \\
        --structures NaCl NbCl5 NaNbO3 --output results/3-T11-<machine>-<date>/mh1-r2scan

Hybrid float32 by default; use an otherwise idle GPU. Output: <output>/metadata.json (every sample's
energies) and a summary on stdout. A structure that does not fit (out of memory) is recorded and skipped.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ab_step_time  # noqa: E402

SCHEMA_VERSION = 1
MOVES = ("swap", "transmute", "displace")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--model", required=True, help="Label or alias (mh1, omol0-xl, ...)")
    p.add_argument("--head", default=None)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5", "NaNbO3"])
    p.add_argument("--superset", nargs="+", default=["Na", "Nb", "O", "Cl"])
    p.add_argument("--temperatures-K", nargs="+", type=float, default=[300.0, 1500.0])
    p.add_argument("--md-steps", type=int, default=2000, help="NVT steps before each snapshot")
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--min-mc-temperature-K", type=float, default=250.0)
    p.add_argument("--max-displacement-A", type=float, default=0.15)
    p.add_argument("--swap-samples", type=int, default=4)
    p.add_argument("--transmute-samples", type=int, default=2)
    p.add_argument("--displace-samples", type=int, default=8)
    p.add_argument("--backend", choices=("auto", "hybrid", "cueq"), default="hybrid")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32", help="force precision")
    p.add_argument("--compile-mode", default=None,
                   help="EngineConfig.compile_mode ('none': eager, still captured; for models whose compile does "
                        "not fit a 15-minute slot, such as OMOL-0 XL)")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args(argv)


def summary(rows, threshold_eV):
    from fastmace.planner import selfcheck

    verdict = selfcheck.move_verdict(rows, threshold_eV)
    out = {}
    for move in MOVES:
        mine = [row for row in rows if row["move"] == move]
        if not mine:
            continue
        entry = dict(verdict["moves"].get(move, {}))
        entry["median_abs_dE_ref_eV"] = sorted(abs(row["reference"]) for row in mine)[len(mine) // 2]
        out[move] = entry
    return {"passed": verdict["passed"], "nonfinite": verdict["nonfinite"], "moves": out}


def free(torch):
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None):
    args = parse_args(argv)
    from fastmace import config
    from fastmace.planner.minimal import MinimalPlanner

    # The allocator first: nothing may touch CUDA before it (the engine's process constraints).
    engine_options = {"backend": args.backend}
    if args.compile_mode is not None:
        engine_options["compile_mode"] = args.compile_mode
    planner = MinimalPlanner(engine=config.EngineConfig(**engine_options))
    hardware = planner.hardware()
    from fastmace.model import loading

    oeq_runtime = str(loading.prepare_hybrid()) if args.backend == "hybrid" else None
    if args.backend == "auto":
        oeq_runtime = str(loading.prepare_hybrid_if_possible())
    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    import torch
    from ase.data import atomic_numbers

    from fastmace.model.forcefield import ForceField
    from fastmace.planner import selfcheck
    from fastmace.planner.workload import Workload

    precision = args.precision
    superset = tuple(atomic_numbers[symbol] for symbol in args.superset)
    samples = {"swap": args.swap_samples, "transmute": args.transmute_samples, "displace": args.displace_samples}
    threshold = selfcheck.move_threshold(args.min_mc_temperature_K)
    meta = dict(schema_version=SCHEMA_VERSION, status="running", tool="move_energy_precision",
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename, python=sys.executable,
                fastmace_commit=ab_step_time.fastmace_commit() or os.environ.get("FASTMACE_SOURCE_COMMIT"),
                compile_mode=args.compile_mode, model=args.model, head=args.head,
                checkpoint=str(checkpoint), checkpoint_sha256=ab_step_time._sha256(checkpoint), backend=args.backend,
                dtype=precision, superset=list(args.superset), temperatures_K=args.temperatures_K,
                md_steps=args.md_steps, timestep_fs=args.timestep_fs, min_mc_temperature_K=args.min_mc_temperature_K,
                threshold_eV=threshold, max_displacement_A=args.max_displacement_A, samples=samples, seed=args.seed,
                oeq_runtime=oeq_runtime, gpu=torch.cuda.get_device_name(0), torch=torch.__version__, runs={})
    path = args.output / "metadata.json"

    def save():
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    try:
        loaded = loading.load_pristine(args.model, args.head, checkpoint=str(checkpoint),
                                       sha256=meta["checkpoint_sha256"])
        meta["model_charge_spin"] = [loaded.charge, loaded.spin_multiplicity]
        save()
        workload = Workload(max_temperature_K=max(args.temperatures_K), species_changes=True, mc_moves=MOVES,
                            min_mc_temperature_K=args.min_mc_temperature_K,
                            max_displacement_A=args.max_displacement_A)
        for spec in args.structures:
            name, base = ab_step_time.read_structure(spec, None)
            engine = None
            try:
                force_field = ForceField(loaded, base, precision=precision, species=superset)
                integrator = config.IntegratorConfig(seed=args.seed, temperature_K=args.temperatures_K[0],
                                                     timestep_fs=args.timestep_fs)
                plan = planner.plan(force_field, base, integrator, workload, hardware)
                meta["runs"][name] = {"stage": "build", "plan_backend": plan.engine.backend}
                save()
                started = time.perf_counter()
                engine = planner.build(plan, force_field, base.copy(), temperature_K=args.temperatures_K[0])
                build_s = time.perf_counter() - started
                del meta["runs"][name]
                for temperature in args.temperatures_K:
                    key = f"{name}@{temperature:g}K"
                    entry = meta["runs"][key] = dict(structure=name, natoms=len(base), temperature_K=temperature,
                                                     build_s=build_s, plan_backend=plan.engine.backend,
                                                     describe=plan.describe())
                    started = time.perf_counter()
                    engine.integrator.set_temperature(temperature)
                    engine.run(args.md_steps)
                    engine.flush()
                    entry["md_s"] = time.perf_counter() - started
                    # A canonical point: the neighbour list rebuilt at the snapshot, so the displacement samples
                    # need no trial rebuild (the gate itself runs right after a build).
                    engine.canonicalise(reevaluate=True)
                    snapshot = base.copy()
                    snapshot.positions = engine.state.positions.detach().to(torch.float64).cpu().numpy()
                    started = time.perf_counter()
                    references = selfcheck.move_references(plan, force_field, snapshot, workload, seed=args.seed,
                                                           samples=samples)
                    entry["references_s"] = time.perf_counter() - started
                    entry["references"] = dict(ran=references.ran, device=references.device,
                                               steps=list(references.steps), unavailable=references.unavailable)
                    entry["configurations"] = len(references.moves.configurations)
                    entry["not_sampleable"] = list(references.moves.not_sampleable)
                    save()
                    if references.unavailable is not None:
                        continue
                    for side in ("float32", "float64") if precision == "float32" else ("float64",):
                        started = time.perf_counter()
                        evaluator = selfcheck._move_evaluator(engine, side, plan, force_field)
                        rows, skipped = selfcheck.move_rows(engine, evaluator, references)
                        entry[side] = dict(seconds=time.perf_counter() - started, info=evaluator.info(),
                                           skipped=skipped, rows=rows, **summary(rows, threshold))
                        save()
            except Exception as error:  # out of memory and the like: recorded, the next structure runs
                meta["runs"].setdefault(f"{name}", {})["error"] = repr(error)
                save()
            finally:
                del engine
                free(torch)
        meta["status"] = "done"
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    table = {key: {side: {move: round(entry[side]["moves"][move]["worst_k_B_T_min"], 6)
                          for move in entry[side]["moves"]} for side in ("float32", "float64") if side in entry}
             for key, entry in meta["runs"].items() if "error" not in entry}
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
