"""MC evaluation cost as a fraction of one MD step (plan tasks 3-T07, then 3-T09, 3-T10 and 3-T20; SPEC §10.4
"Cost", §16 row 3 acceptance item 2).

3-T07's rows: the energy-only evaluation of the committed state (``fastmace.mc.evaluator.MCEvaluator``)
in float32 (the force field's own forward) and float64 (the float64 copy of the same plan), against one
MD step of the same engine. One process, one structure: the engine is planned and built as
``Simulation`` builds it, warmed up, then timed blocks of each side alternate, the order rotating and
mirrored every round. Recorded: per-step and per-evaluation times, energy-only over MD per round,
the setup time of each evaluator (its first call compiles or captures), the float64 copy's memory and
backend, and the energies of both precisions against the MD energy.

    PYTHONPATH=$PWD FASTMACE_STRUCTURES=~/fastmace-baseline/1a-T18 python tools/bench/mc_cost.py \\
        --checkpoint ~/models/mace-mh-1.model --structure NaCl --output results/3-T07-<machine>-<date>/NaCl

Hybrid float32 by default. Use an otherwise idle GPU. Output: <output>/metadata.json. Later tasks add
their rows (swap and transmutation attempts: 3-T09; displacements with and without a trial rebuild:
3-T10) as further sides.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ab_step_time  # noqa: E402

from fastmace.bench import step_time  # noqa: E402

COPY_BACKENDS = ("plan", "cueq")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    p.add_argument("--copy-backend", choices=COPY_BACKENDS, default="plan",
                   help="backend of the float64 copy: the plan's, falling back to cuEq if it fails, or cuEq")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=100)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args(argv)


class EnergySide:
    """An ``MCEvaluator`` as a ``fastmace.bench.step_time.TimedEngine``: one "step" is one energy evaluation."""

    def __init__(self, evaluator, torch):
        self.evaluator, self.torch = evaluator, torch

    def step(self):
        self.evaluator.evaluate()

    def prepare_block(self):
        pass

    def finish_block(self):
        pass

    def synchronize(self):
        self.torch.cuda.synchronize()


def round_order(names, index):
    """Each round rotates the sides by one and every second round is mirrored (no side always first)."""
    shift = index % len(names)
    order = list(names[shift:]) + list(names[:shift])
    return order[::-1] if (index // len(names)) % 2 else order


def summarise(blocks, names):
    out = {name: step_time.BlockStatistics.from_blocks(blocks[name]).as_dict() for name in names}
    md = blocks["md"]
    for name in names:
        if name != "md":
            ratios = [e / m for e, m in zip(blocks[name], md)]
            out[name]["over_md_per_round"] = ratios
            out[name]["over_md_median"] = statistics.median(ratios)
    return out


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model import loading
    oeq_runtime = str(loading.prepare_hybrid()) if args.backend == "hybrid" else None
    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    import torch

    from fastmace import config
    from fastmace.mc.evaluator import MCEvaluator, float64_model
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.minimal import MinimalPlanner
    from fastmace.planner.workload import Workload

    precision = "float32" if args.backend == "hybrid" else "float64"
    meta = dict(schema_version=step_time.SCHEMA_VERSION, status="running", tool="mc_cost",
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename, python=sys.executable,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, checkpoint=str(checkpoint),
                checkpoint_sha256=ab_step_time._sha256(checkpoint), backend=args.backend, dtype=precision,
                temperature_K=args.temperature_K, block=args.block, rounds=args.rounds, warmup=args.warmup,
                oeq_runtime=oeq_runtime, gpu=torch.cuda.get_device_name(0), torch=torch.__version__, sides={})
    path = args.output / "metadata.json"

    def save():
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    try:
        planner = MinimalPlanner(engine=config.EngineConfig(backend=args.backend))
        hardware = planner.hardware()  # sets the allocator before the first CUDA allocation
        loaded = loading.load_pristine(args.model, checkpoint=str(checkpoint), sha256=meta["checkpoint_sha256"])
        name, base = ab_step_time.read_structure(args.structure, None)
        meta.update(structure=name, natoms=len(base))
        force_field = ForceField(loaded, base, precision=precision)
        integrator = config.IntegratorConfig(seed=args.seed, temperature_K=args.temperature_K)
        plan = planner.plan(force_field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
        started = time.perf_counter()
        engine = planner.build(plan, force_field, base.copy(), temperature_K=args.temperature_K)
        md = ab_step_time.FastMACESide(engine, torch)
        step_time.warm_up(md, args.warmup)
        meta["sides"]["md"] = dict(setup_s=time.perf_counter() - started, describe=plan.describe())
        sides = {"md": md}
        evaluators = {}

        def add(side, make):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            evaluator = make()
            built_s = time.perf_counter() - started
            first = evaluator.evaluate()  # compiles or captures
            torch.cuda.synchronize()
            first_s = time.perf_counter() - started - built_s
            evaluators[side] = evaluator
            sides[side] = EnergySide(evaluator, torch)
            step_time.warm_up(sides[side], 10)
            meta["sides"][side] = dict(
                build_s=built_s, first_call_s=first_s, kept_bytes=torch.cuda.memory_allocated() - before,
                peak_bytes_during_setup=torch.cuda.max_memory_allocated() - before, info=evaluator.info(),
                energy_eV=float(first[0]))
            save()

        add("energy_" + precision, lambda: MCEvaluator(engine))
        if precision == "float32":
            def copy():
                backends = [args.backend, "cueq"] if args.copy_backend == "plan" else ["cueq"]
                errors = {}
                for backend in dict.fromkeys(backends):
                    try:
                        evaluator = MCEvaluator(engine, "float64",
                                                model=float64_model(force_field, plan, backend=backend))
                        evaluator.evaluate()
                        meta["copy_backend"], meta["copy_backend_errors"] = backend, errors
                        return evaluator
                    except Exception as error:  # record and fall back (conductor answer 1)
                        errors[backend] = repr(error)
                raise RuntimeError(f"no float64 copy: {errors}")

            add("energy_float64", copy)
        names = tuple(sides)
        blocks = {name: [] for name in names}
        for index in range(args.rounds):
            for name in round_order(names, index):
                blocks[name].append(step_time.time_block(sides[name], args.block))
        meta["timing"] = summarise(blocks, names)
        md.finish_block()
        md_energy = float(engine.state.energy[0])
        meta["energies_after_md"] = {"md_eV": md_energy, **{
            side: float(evaluator.baseline()[0]) for side, evaluator in evaluators.items()}}
        meta["status"] = "done"
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    print(json.dumps({side: dict(ms=value["median_ms"], over_md=value.get("over_md_median"))
                      for side, value in meta.get("timing", {}).items()}, indent=1))


if __name__ == "__main__":
    main()
