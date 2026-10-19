"""The three front ends in one process: ``Simulation``, ``fm.md`` and ``fastmace.ase.Langevin`` (1b-T23, item 4).

SPEC §7.7's acceptance for the adoption layer: for the same inputs the three give the same plan and the
same gate record, and their steady-state step times are equal within the noise floor (1a-T18: 0.16 %
for 12 blocks). This tool builds one side per front end on one structure (MH-1 NaCl 3x3x3 by
default), Langevin NVT at --temperature-K with ``fm.md``'s defaults (timestep 0.5 fs, friction 0.02 per
ASE time unit, main.py's), the same seed and the same ``fm.Output`` cadences in a directory per side.
Then it warms each side up and times blocks round-robin (A B C A B C ...), each block one ``run(n)`` of
its front end: ``Simulation.run(steps=n)``, ``RunResult.simulation.run(steps=n)`` after ``fm.md``'s
first step, and ``Langevin.run(n)``. Ratios are against the first side.

All three build their engine through the minimal planner in this process, so they share one backend
family (§8.7) and need no patch handling. ``fm.md`` arrives with 1b-T20 and ``fastmace.ase.Langevin``
with 1b-T21; until then those sides stop with the task they wait for. Use an otherwise idle GPU.
Output: <output>/metadata.json, and each side's run directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
from fastmace.bench import step_time

FRONT_ENDS = ("simulation", "md", "ase")
WAITS_FOR = {"md": "1b-T20 (fm.md)", "ase": "1b-T21 (fastmace.ase.Langevin)"}
# fm.md's defaults (1b-T20): 0.5 fs, main.py's friction 0.02 per ASE time unit.
TIMESTEP_FS = 0.5
FRICTION_ASE = 0.02


class FrontEndUnavailable(RuntimeError):
    """A front end this tool times is not in this checkout yet."""


def require_front_end(name):
    """Raise ``FrontEndUnavailable`` naming the plan task when front end ``name`` is still a stub or absent."""
    if name == "simulation":
        return
    if name == "md":
        from fastmace import quick

        source = inspect.getsource(quick.md)
        if "raise NotImplementedError" in source:
            raise FrontEndUnavailable(f"fm.md is a stub: this side waits for {WAITS_FOR['md']}")
        return
    if name == "ase":
        if importlib.util.find_spec("fastmace.ase.md") is None:
            raise FrontEndUnavailable(f"fastmace.ase.Langevin is missing: this side waits for {WAITS_FOR['ase']}")
        return
    raise ValueError(f"unknown front end {name!r}; the front ends are {FRONT_ENDS}")


class ChunkSide:
    """A front end as a ``step_time.TimedEngine``: the steps of a block go to one ``run(n)`` call.

    ``step_time.time_block`` times ``prepare_block``, the steps, ``finish_block`` and ``synchronize``
    together, so a block's time is that of ``runner(block)``, the way a user runs the front end.
    """

    def __init__(self, runner: Callable[[int], object], synchronize: Callable[[], object]):
        self.runner, self._synchronize = runner, synchronize
        self.pending = 0

    def step(self):
        self.pending += 1

    def prepare_block(self):
        pass

    def finish_block(self):
        if self.pending:
            count, self.pending = self.pending, 0
            self.runner(count)

    def synchronize(self):
        self._synchronize()


def run_round_robin(sides: Mapping[str, object], protocol: step_time.TimingProtocol,
                    clock: Callable[[], float] = time.perf_counter) -> dict:
    """Warm up every side, then time ``protocol.blocks`` rounds of one block per side in order."""
    names = list(sides)
    if len(names) < 2:
        raise ValueError(f"a comparison needs at least two sides, not {names}")
    for name in names:
        step_time.warm_up(sides[name], protocol.warmup)
    blocks = {name: [] for name in names}
    order = []
    for _ in range(protocol.blocks):
        for name in names:
            blocks[name].append(step_time.time_block(sides[name], protocol.block, clock))
            order.append(name)
    statistics = {name: step_time.BlockStatistics.from_blocks(blocks[name]) for name in names}
    first = statistics[names[0]].median_ms
    return dict(order=order, sides={name: statistics[name].as_dict() for name in names},
                ratios_over_first={name: statistics[name].median_ms / first for name in names[1:]},
                statistics=statistics)


def plans_equal(first, second) -> bool:
    """``Plan`` equality, with the Langevin friction allowed to differ by one ulp (T21's conversion from ASE's)."""
    if first == second:
        return True
    a, b = first.integrator, second.integrator
    if a is None or b is None:
        return False
    if abs(a.friction - b.friction) > math.ulp(max(abs(a.friction), abs(b.friction))):
        return False
    return first == dataclasses.replace(second, integrator=dataclasses.replace(b, friction=a.friction))


def gate_records_equal(first, second) -> bool:
    """Gate records equal except their timings (``seconds``)."""
    def strip(record):
        data = record.to_metadata()
        data.pop("seconds", None)
        return data

    return strip(first) == strip(second)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--sides", nargs="+", choices=FRONT_ENDS, default=list(FRONT_ENDS),
                   help="Front ends in build and block order; ratios are against the first")
    p.add_argument("--model", default="mh1-r2scan", help="Label (its file under FASTMACE_MODELS)")
    p.add_argument("--structure", default="NaCl",
                   help="Benchmark name (under FASTMACE_STRUCTURES) or a file with PATH@NXxNYxNZ")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--check", choices=("skip", "gate"), default="skip", help="The gate (opt-in, owner decision D1)")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    args = p.parse_args(argv)
    if len(args.sides) < 2 or len(set(args.sides)) != len(args.sides):
        p.error("--sides takes two or three distinct front ends")
    return args


# ------------------------------------------------------------------ the sides


def _output(args, name):
    from fastmace.io.output import Output

    # The ASE shim writes thermo rows and restarts at the production cadences and its trajectory itself (none
    # here): every side gets the same Output, without a FastMACE trajectory.
    return Output(str(args.output / name), trajectory_every=None)


def build_simulation(args, atoms, loaded):
    from ase import units

    from fastmace import integrators
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation
    integrator = integrators.Langevin(timestep_fs=TIMESTEP_FS, temperature_K=args.temperature_K,
                                      friction_per_fs=FRICTION_ASE * units.fs)
    sim = Simulation(atoms, ForceField(loaded, atoms), integrator=integrator, output=_output(args, "simulation"),
                     seed=args.seed, check=args.check)
    return sim, lambda steps: sim.run(steps=steps), sim.plan, sim.gate


def build_md(args, atoms, loaded):
    import fastmace as fm

    result = fm.md(atoms, model=loaded, temperature_K=args.temperature_K, timestep_fs=TIMESTEP_FS, steps=1,
                   seed=args.seed, check=args.check, output=_output(args, "md"))
    sim = result.simulation
    return result, lambda steps: sim.run(steps=steps), result.plan, result.gate


def build_ase(args, atoms, loaded):
    from ase import units

    from fastmace.ase import Langevin
    dyn = Langevin(atoms.copy(), TIMESTEP_FS * units.fs, temperature_K=args.temperature_K, friction=FRICTION_ASE,
                   model=loaded, seed=args.seed, check=args.check, output=_output(args, "ase").directory)
    dyn.run(1)  # the shim builds its Simulation at the first run, as fm.md runs its first step
    return dyn, dyn.run, dyn.plan, dyn.gate


BUILDERS = {"simulation": build_simulation, "md": build_md, "ase": build_ase}


def main(argv=None):
    args = parse_args(argv)
    for name in args.sides:
        require_front_end(name)
    from fastmace.model import loading

    loaded = loading.load_pristine(args.model)  # prepares the OEq runtime before MACE is imported (1b-T08b)
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ab_step_time
    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    structure, atoms = ab_step_time.read_structure(args.structure, None)
    meta = dict(schema_version=step_time.SCHEMA_VERSION, status="running", tool="ab_frontends",
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename, python=sys.executable,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, sha256=loaded.sha256,
                structure=structure, natoms=len(atoms), temperature_K=args.temperature_K, check=args.check,
                timestep_fs=TIMESTEP_FS, friction_ase=FRICTION_ASE, protocol=protocol.as_dict(), sides=args.sides,
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__, setup_s={}, plans={}, gates={})
    path = args.output / "metadata.json"

    def save():
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    try:
        sides, plans, gates, owners = {}, {}, {}, {}
        for name in args.sides:
            started = time.perf_counter()
            owners[name], runner, plans[name], gates[name] = BUILDERS[name](args, atoms, loaded)
            meta["setup_s"][name] = time.perf_counter() - started
            sides[name] = ChunkSide(runner, torch.cuda.synchronize)
            meta["plans"][name] = plans[name].describe() if plans[name] is not None else None
            meta["gates"][name] = gates[name].to_metadata() if gates[name] is not None else None
            save()
        first = args.sides[0]
        meta["same_plan"] = {name: plans[name] is not None and plans_equal(plans[first], plans[name])
                             for name in args.sides[1:]}
        meta["same_gate"] = {name: gates[name] is not None and gate_records_equal(gates[first], gates[name])
                             for name in args.sides[1:]}
        result = run_round_robin(sides, protocol)
        result.pop("statistics")
        meta["timing"] = result
        meta["status"] = "complete"
        print(" ".join(f"{name} {result['sides'][name]['median_ms']:.3f} ms" for name in args.sides),
              "ratios", json.dumps(result["ratios_over_first"]), "same plan", json.dumps(meta["same_plan"]),
              flush=True)
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return meta


if __name__ == "__main__":
    main()
