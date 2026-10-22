"""3-T15's local GPU smoke: ``fm.md`` with a swap block, then a restart and a resume (SPEC §7.6, §7.7).

    PYTHONPATH=$PWD FASTMACE_STRUCTURES=~/fastmace-baseline/1a-T18 python tools/validation/mc_front_end_smoke.py \\
        --checkpoint ~/models/mace-mh-1.model --run ~/fastmace-runs/3-T15-smoke \\
        --output results/3-T15-<machine>-<date>/NaCl --commit <fastmace commit>

The run, on the default planner (hybrid where OEq loads):
1. ``fm.md`` on NaCl 3x3x3, NVT 0.5 fs at ``--temperature-K``, ``--steps`` steps, with
   ``Swap(Na-Cl, every=--every, attempts=--attempts)``; restart every ``--every * 3`` steps;
2. ``Simulation.resume`` of that run directory, then ``--resume-steps`` more steps: the move set
   stays attached across the resume (§7.6), and its counters continue.

It records in ``<output>/smoke.json``: the plan's tier and backend, both reports, ``counters.mc``
before and after the resume, the blocks, the thermo MC columns of the last row, and the checks
below, each true or false. The run directory (trajectory, thermo, restart) stays outside the
repository.

- the move set ran: ``counters.mc.swap.attempted`` is blocks x attempts;
- the cost line shows a device time per attempt (CUDA events, first block left out);
- the thermo columns equal the counters;
- after the resume the attempts continue from the restart's counters.
"""

from __future__ import annotations

import fastmace  # noqa: F401  (before torch: file locks, SPEC §14)

# isort: split
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/, for ab_step_time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--run", required=True, type=Path, help="fresh run directory, outside the repository")
    p.add_argument("--output", required=True, type=Path, help="fresh results directory")
    p.add_argument("--commit", default=None, help="the FastMACE commit (WSL git cannot read a worktree's .git)")
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--resume-steps", type=int, default=600)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--attempts", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model import loading

    hybrid = loading.prepare_hybrid_if_possible()  # before anything imports MACE (the hybrid plan needs it)
    import ab_step_time
    import torch

    from fastmace import mc
    from fastmace.io import output as output_io
    from fastmace.io.output import Output
    from fastmace.quick import md
    from fastmace.simulation import Simulation

    args.output.mkdir(parents=True, exist_ok=False)
    run = args.run.expanduser()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    record = dict(tool="mc_front_end_smoke", created_utc=datetime.now(timezone.utc).isoformat(),
                  host=os.uname().nodename, python=sys.executable, fastmace_commit=args.commit,
                  model=args.model, checkpoint=str(checkpoint), checkpoint_sha256=ab_step_time._sha256(checkpoint),
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__, temperature_K=args.temperature_K,
                  steps=args.steps, resume_steps=args.resume_steps, every=args.every, attempts=args.attempts,
                  seed=args.seed, oeq_runtime=hybrid, status="running")
    path = args.output / "smoke.json"

    def save():
        path.write_text(json.dumps(record, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")

    save()
    try:
        loaded = loading.load_pristine(args.model, checkpoint=str(checkpoint), sha256=record["checkpoint_sha256"])
        name, atoms = ab_step_time.read_structure(args.structure, None)
        record.update(structure=name, natoms=len(atoms))
        swap = mc.Swap(pairs=[("Na", "Cl")], every=args.every, attempts=args.attempts)
        output = Output(str(run), trajectory_every=None, thermo_every=args.every // 2, restart_every=args.every * 3)
        started = time.perf_counter()
        result = md(atoms, model=loaded, temperature_K=args.temperature_K, steps=args.steps, mc=swap,
                    output=output, seed=args.seed)
        record["md_s"] = time.perf_counter() - started
        sim = result.simulation
        plan = sim.plan
        record.update(tier=plan.tier, backend=plan.engine.backend if plan.engine is not None else None,
                      report=result.summary, counters_mc=result.metadata["counters"]["mc"],
                      report_mc=result.metadata["report"]["mc"])
        restart_step = (args.steps // (args.every * 3)) * args.every * 3
        sim.close()
        blocks = args.steps // args.every
        attempted = record["counters_mc"]["swap"]["attempted"]
        header, rows = output_io.read_thermo(run)
        last = dict(zip(header, rows[-1]))
        record["thermo_last"] = last
        checks = {
            "attempted_is_blocks_times_attempts": attempted == blocks * args.attempts,
            "cost_line_has_device_time": "ms/attempt" in result.summary,
            "thermo_equals_counters": int(last["mc_swap_attempted"]) == attempted
            and int(last["mc_swap_accepted"]) == record["counters_mc"]["swap"]["accepted"],
        }
        save()

        started = time.perf_counter()
        resumed = Simulation.resume(run)
        record["resume_s"] = time.perf_counter() - started
        before = resumed.metadata()["counters"]["mc"]
        resumed.run(steps=args.resume_steps)
        resumed.close()
        after = resumed.metadata()["counters"]["mc"]
        record.update(restart_step=restart_step, counters_mc_at_resume=before, counters_mc_after_resume=after,
                      report_after_resume=resumed.report.text())
        restart_blocks = restart_step // args.every
        resume_blocks = (restart_step + args.resume_steps) // args.every - restart_blocks
        checks["resume_continues_the_counters"] = (
            before["swap"]["attempted"] == restart_blocks * args.attempts
            and after["swap"]["attempted"] == (restart_blocks + resume_blocks) * args.attempts)
        record.update(checks=checks, status="passed" if all(checks.values()) else "failed")
    except Exception as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise
    save()
    print(json.dumps({"status": record["status"], "checks": record.get("checks")}, indent=2))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
