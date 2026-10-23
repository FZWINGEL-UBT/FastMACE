"""3b-T05: BAOAB's per-step copy of the pressure target changes no numbers, and what it costs (GPU).

The double buffer of 3b-T05 adds one device copy to BAOAB's stage A (``externalstress_step.copy_(
externalstress)``), which stage B reads instead of the filled target. At constant pressure the copy holds
the same value, so a trajectory must be bit for bit the one of the class before the change. This probe
loads that class from git (``--before``, default ``main``: ``fastmace/integrators/baoab.py`` as committed
there; or ``--before-source``, that file exported beforehand, with ``--before-commit`` and ``--fastmace-commit``
where git cannot read the checkout, as WSL in a Windows worktree) next to the current one, in one process,
and runs both on the GPU:

1. **Bitwise:** checked and pipelined engines with captured graphs, hydrostatic and full cell, constant
   0.5 GPa, a temperature ramp, ``--steps`` steps; every output array compared at four points.
2. **Cost:** two pipelined hydrostatic engines (after, before), warmed up, then ``--blocks`` alternating
   timed blocks of ``--block-steps`` steps each (CUDA-synchronised wall time); the per-step difference
   of the block medians and its spread.

The force field is ``tests/gpu/test_engine_gpu.py``'s deterministic Lennard-Jones (one-hot sums, no
atomics), so runs are reproducible on the GPU; the copy is independent of the force field.

    PYTHONPATH=$PWD python tools/validation/baoab_target_copy.py --output results/<run-id>
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def class_before(ref, source_path=None):
    """``LangevinBAOAB`` as committed at ``ref`` (or read from ``source_path``), as a sibling module."""
    source = git("show", f"{ref}:fastmace/integrators/baoab.py") if source_path is None else source_path.read_text()
    module = types.ModuleType("fastmace.integrators.baoab_before")
    module.__package__ = "fastmace.integrators"
    module.__file__ = f"<{ref}:fastmace/integrators/baoab.py>"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.LangevinBAOAB


def build(cls, mode, cell_mode):
    import torch  # noqa: F401  (fastmace first, SPEC §14)
    from test_engine_gpu import IN_PROCESS, DeterministicLJ, argon

    from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
    from fastmace.engine import constraints
    from fastmace.engine.graph import GraphEngine
    from fastmace.state import SimulationState

    atoms = argon()
    neighbours = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="on",
                                 edge_bucket_fraction=0.0, edge_multiple=64)
    field = DeterministicLJ(atoms, neighbours, compute_stress=True)
    state = SimulationState.from_atoms(atoms, "cuda", 7, compute_stress=True)
    dyn = cls(state, IntegratorConfig(ensemble="npt", timestep_fs=2.0, friction=0.02, noise_batch=7,
                                      pressure_GPa=0.5, barostat_mass_factor=0.01, cell_mode=cell_mode))
    config = EngineConfig(device="cuda:0", pipelined="on" if mode == "pipelined" else "off", **IN_PROCESS)
    return GraphEngine(config, field, dyn, state, capture=True, pad_edges=True,
                       constraints=constraints.ProcessRecord())


def run(engine, steps, every):
    rows = []
    for step in range(1, steps + 1):
        engine.integrator.set_temperature(150.0 * step / steps + 30.0)
        engine.step()
        if step % every == 0:
            rows.append(engine.sync())
    return rows


def bitwise(after, before, steps):
    import numpy as np

    out = {}
    for mode in ("checked", "pipelined"):
        for cell_mode in ("hydrostatic", "full"):
            rows_after = run(build(after, mode, cell_mode), steps, steps // 4)
            rows_before = run(build(before, mode, cell_mode), steps, steps // 4)
            worst = 0.0
            equal = True
            for got, want in zip(rows_after, rows_before, strict=True):
                for key in want:
                    same = np.array_equal(got[key], want[key])
                    equal &= same
                    if not same:
                        worst = max(worst, float(np.abs(got[key] - want[key]).max()))
            out[f"{mode}/{cell_mode}"] = {"bitwise_equal": bool(equal), "max_abs_difference": worst,
                                          "final_cell_diagonal": [float(x) for x in np.diag(rows_after[-1]["cell"])]}
    return out


def cost(after, before, blocks, block_steps, warmup):
    import torch

    engines = {"after": build(after, "pipelined", "hydrostatic"), "before": build(before, "pipelined", "hydrostatic")}
    for engine in engines.values():
        engine.integrator.set_temperature(100.0)
        engine.run(warmup)
        engine.flush()
    torch.cuda.synchronize()
    times = {name: [] for name in engines}
    for block in range(blocks):
        order = ("after", "before") if block % 2 == 0 else ("before", "after")
        for name in order:
            engine = engines[name]
            torch.cuda.synchronize()
            start = time.perf_counter()
            engine.run(block_steps)
            engine.flush()
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - start) / block_steps * 1e6)  # us per step
    differences = [a - b for a, b in zip(times["after"], times["before"], strict=True)]
    return {"us_per_step": {name: values for name, values in times.items()},
            "median_us_per_step": {name: statistics.median(values) for name, values in times.items()},
            "median_difference_us": statistics.median(differences),
            "difference_range_us": [min(differences), max(differences)],
            "blocks": blocks, "block_steps": block_steps, "warmup_steps": warmup,
            "rebuilds": {name: int(engine.force_field.nl.rebuilds) for name, engine in engines.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", default="main")
    parser.add_argument("--before-source", type=Path, help="the before ref's baoab.py, exported (no git here)")
    parser.add_argument("--before-commit", help="the before ref's commit, with --before-source")
    parser.add_argument("--fastmace-commit", help="this checkout's commit, where git cannot read it")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "tests" / "gpu"), str(ROOT / "tests")]

    import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

    # isort: split
    import torch

    from fastmace.integrators.baoab import LangevinBAOAB

    before = class_before(args.before, args.before_source)
    record = {
        "probe": "tools/validation/baoab_target_copy.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "python": sys.executable,
        "torch": torch.__version__,
        "fastmace_commit": args.fastmace_commit or git("rev-parse", "HEAD"),
        "before_ref": args.before, "before_commit": args.before_commit or git("rev-parse", args.before),
        "force_field": "tests/gpu/test_engine_gpu.py DeterministicLJ, argon 3x3x3 (108 atoms), float64",
        "bitwise": bitwise(LangevinBAOAB, before, args.steps),
        "cost": cost(LangevinBAOAB, before, args.blocks, args.block_steps, args.warmup),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baoab_target_copy.json").write_text(json.dumps(record, indent=2) + "\n")
    summary = {key: record[key] for key in ("gpu", "fastmace_commit", "before_commit")}
    summary["bitwise"] = {key: value["bitwise_equal"] for key, value in record["bitwise"].items()}
    summary["cost"] = {key: record["cost"][key] for key in ("median_us_per_step", "median_difference_us",
                                                             "difference_range_us")}
    print(json.dumps(summary, indent=2))
    return 0 if all(summary["bitwise"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
