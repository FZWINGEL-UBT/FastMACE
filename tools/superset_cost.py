"""The cost of a species superset larger than the structure (plan task 3-T02; SPEC §8.5, §18 risk row).

Two sides in one process, on one structure: ``structure`` is a ``ForceField`` whose superset is the
structure's species (the default plan), ``superset`` one with ``--species`` declared (default
Na Cl Nb O on NaCl). Both go through the minimal planner (what ``Simulation`` builds), are built and
warmed up, and their timed blocks then alternate (``fastmace.bench.ab.run_ab``, both orders with
``--pattern abba``). Recorded per side: the step time, the GPU memory the side's build kept
(allocated after its build and warm-up, minus before), the peak during its warm-up, and the
reassociation's slot count. The superset side has its own compiled model (the compile key has the
superset's columns), so the two never share one.

    PYTHONPATH=$PWD FASTMACE_STRUCTURES=~/fastmace-baseline/1a-T18 python tools/superset_cost.py \\
        --checkpoint ~/models/mace-mh-1.model --output results/3-T02-<machine>-<date>/NaCl

Hybrid float32 by default (no launcher needed: FastMACE prepares its OEq runtime in-process). Use an
otherwise idle GPU. Output: <output>/metadata.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402

from fastmace.bench import ab, step_time  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--species", nargs="+", default=["Na", "Cl", "Nb", "O"])
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--pattern", choices=ab.PATTERNS, default="abba")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args(argv)


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    args = parse_args(argv)
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    from fastmace.model import compile as compiling
    from fastmace.model import loading
    allocator = compiling.configure_allocator()  # before the first CUDA allocation (the engine checks it)
    oeq_runtime = str(loading.prepare_hybrid()) if args.backend == "hybrid" else None
    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    import torch

    from fastmace import config
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.minimal import MinimalPlanner
    from fastmace.planner.workload import Workload

    meta = dict(schema_version=step_time.SCHEMA_VERSION, status="running", tool="superset_cost",
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename, python=sys.executable,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, checkpoint=str(checkpoint),
                checkpoint_sha256=_sha256(checkpoint), backend=args.backend,
                dtype="float32" if args.backend == "hybrid" else "float64", temperature_K=args.temperature_K,
                protocol=protocol.as_dict(), pattern=args.pattern, oeq_runtime=oeq_runtime, cuda_allocator=allocator,
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__, sides={})
    path = args.output / "metadata.json"

    def save():
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    try:
        planner = MinimalPlanner(engine=config.EngineConfig(backend=args.backend))
        hardware = planner.hardware()
        loaded = loading.load_pristine(args.model, checkpoint=str(checkpoint), sha256=meta["checkpoint_sha256"])
        name, base = ab_step_time.read_structure(args.structure, None)
        meta.update(structure=name, natoms=len(base), structure_species=sorted({int(z) for z in base.numbers}))
        integrator = config.IntegratorConfig(seed=args.seed, temperature_K=args.temperature_K)
        precision = "float32" if args.backend == "hybrid" else "float64"
        sides = {}
        for side, species in (("structure", None), ("superset", args.species)):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            started = time.perf_counter()
            force_field = ForceField(loaded, base, precision=precision, species=species)
            plan = planner.plan(force_field, base, integrator, Workload(max_temperature_K=args.temperature_K),
                                hardware)
            engine = planner.build(plan, force_field, base.copy(), temperature_K=args.temperature_K)
            setup_s = time.perf_counter() - started
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timed = ab_step_time.FastMACESide(engine, torch)
            step_time.warm_up(timed, protocol.warmup)
            field = engine.force_field
            reassociation = [i.first_reassociation for i in getattr(getattr(field.model, "_orig_mod", field.model),
                                                                     "interactions", ())
                             if getattr(i, "first_reassociation", None) is not None]
            meta["sides"][side] = dict(
                superset=list(field.superset), species_columns=list(field.species_columns), setup_s=setup_s,
                kept_bytes=torch.cuda.memory_allocated() - before, warmup_peak_bytes=torch.cuda.max_memory_allocated(),
                reassociation_slots=[int(r.columns.numel()) for r in reassociation],
                sender_group_count=field.neighbours.group_count if field.neighbours.sender_groups is not None else None,
                rewrites=[entry.name for entry in field.applied], describe=plan.describe(full=True))
            sides[side] = timed
            save()
        result = ab.run_ab(sides, step_time.TimingProtocol(warmup=0, block=args.block, blocks=args.blocks),
                           pattern=args.pattern)
        meta["ab"] = result.as_dict()
        meta["superset_over_structure"] = result.ratio
        for side, timed in sides.items():
            meta["sides"][side]["observables"] = timed.observables()
        meta["status"] = "done"
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    print(json.dumps(dict(ratio=meta.get("superset_over_structure"),
                          sides={k: dict(kept_MiB=v["kept_bytes"] / 2**20, slots=v["reassociation_slots"])
                                 for k, v in meta["sides"].items()}), indent=1))


if __name__ == "__main__":
    main()
