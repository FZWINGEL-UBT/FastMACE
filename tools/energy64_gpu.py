"""Float64 energy accumulation on the GPU (plan task 1b-T03; SPEC §7.3, §4.3).

Two commands, each in its own process, writing one JSON file:

  energy  the potential energy of one model on one benchmark structure (3 x 3 x 3), as
          --side reference: plain cuEquivariance float64, eager, no rewrites, with the float64 sum
                            and with MACE's own (stock) sum;
          --side production: main.py's fixed plan for --backend (float32 for the hybrid), its
                            initial evaluation with the float64 sum and, with --float64-energy off,
                            with 1a's float32 sum.
          References and production engines run in separate processes (two backend families never
          share one; plan phase-1b "Resource rules").
  ab      steady-state step time of two FastMACE engines in one process, alternating blocks
          (fastmace.bench.ab.run_ab): main.py's plan with the float64 sum against 1a's sum.

Structures are read as tools/ab_step_time.py reads them in a FastMACE-only process (benchmark
names under FASTMACE_STRUCTURES, the baseline export). Use an otherwise idle GPU; ``ab`` refuses
a busy one. docs/validation/1b-T03-energy.md has the commands and the results.
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
import ab_step_time  # noqa: E402  (tools/ab_step_time.py: FastMACESide, read_structure, gpu_processes)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("energy", "ab"))
    p.add_argument("--output", required=True, type=Path, help="JSON file (energy) or fresh directory (ab)")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--checkpoint", type=Path, help="Default: the label's file under FASTMACE_MODELS")
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--side", choices=("reference", "production"), default="production")
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    p.add_argument("--float64-energy", choices=("on", "off"), default="on")
    p.add_argument("--rattle", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--first", choices=("float64_sum", "float32_sum_1a"), default="float64_sum",
                   help="ab: the side built (and timed) first, to control for build order")
    return p.parse_args(argv)


def structure(args):
    name, atoms = ab_step_time.read_structure(args.structure, None)
    if args.rattle:
        atoms.rattle(args.rattle, seed=args.seed)
    return name, atoms


def load_kwargs(args):
    return {} if args.checkpoint is None else {"checkpoint": str(args.checkpoint.expanduser().resolve(strict=True))}


def planner(args, float64_energy):
    from fastmace.config import EngineConfig, PrecisionConfig
    from fastmace.planner.fixed import FixedPlanner

    return FixedPlanner(engine=EngineConfig(backend=args.backend),
                        precision=PrecisionConfig(float64_energy=float64_energy))


def gpu_name():
    import torch

    return torch.cuda.get_device_name(0)


def reference_energies(args, atoms):
    """Plain cuEq float64, eager, no rewrites: (float64 sum, MACE's own sum) in eV."""
    from fastmace.model import loading
    from fastmace.model.forces import MACEForceField

    loaded = loading.load(args.model, backend="cueq", dtype="float64", device="cuda:0", **load_kwargs(args))
    out = {}
    for name, flag in (("float64_sum", True), ("stock_sum", False)):
        field = MACEForceField(loaded, atoms, float64_energy=flag)
        field.evaluate(atoms)  # a throwaway first evaluation
        out[name] = field.evaluate(atoms)[0]
    return out, loaded


def production_energy(args, atoms, flag):
    """main.py's fixed plan for --backend, built: the initial evaluation's energy in eV."""
    from fastmace.config import IntegratorConfig
    from fastmace.engine import constraints

    p = planner(args, flag)
    hardware = p.hardware()
    loaded = p.load(args.model, **load_kwargs(args))
    plan = p.plan(loaded, atoms, IntegratorConfig(seed=args.seed), hardware)
    engine = p.build(plan, loaded, atoms.copy(), temperature_K=args.temperature_K,
                     constraints=constraints.ProcessRecord())
    return engine.potential_energy(), plan, loaded


def command_energy(args):
    if args.backend == "hybrid" and args.side == "production":
        ab_step_time.prepare_fastmace_hybrid()  # before anything imports MACE
    name, atoms = structure(args)
    record = dict(command="energy", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                  gpu=None, python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(), model=args.model,
                  structure=name, natoms=len(atoms), rattle=args.rattle, seed=args.seed, side=args.side)
    start = time.perf_counter()
    if args.side == "reference":
        energies, loaded = reference_energies(args, atoms)
        record.update(backend="cueq", dtype="float64", energies_eV=energies)
    else:
        flag = args.float64_energy == "on"
        value, plan, loaded = production_energy(args, atoms, flag)
        record.update(backend=plan.engine.backend, dtype=plan.precision.dtype, float64_energy=flag,
                      energies_eV={"float64_sum" if flag else "stock_sum": value})
    record.update(gpu=gpu_name(), checkpoint=str(loaded.path), checkpoint_sha256=loaded.sha256,
                  seconds=time.perf_counter() - start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["energies_eV"]))


def command_ab(args):
    from fastmace.bench import ab, step_time

    if args.backend == "hybrid":
        ab_step_time.prepare_fastmace_hybrid()
    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    from fastmace.config import IntegratorConfig
    from fastmace.engine import constraints
    from fastmace.model import compile as compiling

    name, atoms = structure(args)
    record = constraints.ProcessRecord()
    cache = compiling.CompiledModelCache()
    sides, plans = {}, {}
    order = (("float64_sum", True), ("float32_sum_1a", False))
    for side, flag in (order if args.first == "float64_sum" else order[::-1]):
        p = planner(args, flag)
        hardware = p.hardware()
        loaded = p.load(args.model, **load_kwargs(args))
        plan = p.plan(loaded, atoms, IntegratorConfig(seed=args.seed), hardware)
        engine = p.build(plan, loaded, atoms.copy(), temperature_K=args.temperature_K, compile_cache=cache,
                         constraints=record)
        sides[side] = ab_step_time.FastMACESide(engine, torch)
        plans[side] = dict(backend=plan.engine.backend, dtype=plan.precision.dtype, float64_energy=flag)
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    result = ab.run_ab(sides, protocol, pattern="abba")
    meta = dict(command="ab", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                gpu=gpu_name(), python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(),
                model=args.model, structure=name, natoms=len(atoms), protocol=protocol.as_dict(), plans=plans,
                first=args.first,
                result=result.as_dict(), observables={side: s.observables() for side, s in sides.items()})
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    a, b = result.names
    print(f"{name}: {b} / {a} = {result.ratio:.4f} "
          f"({', '.join(f'{r:.4f}' for r in result.pair_ratios)})")


def main(argv=None):
    args = parse_args(argv)
    (command_energy if args.command == "energy" else command_ab)(args)


if __name__ == "__main__":
    main()
