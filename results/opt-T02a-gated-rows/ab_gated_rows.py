"""Same-process A/B of OPT-T02 stage A: ``gated_rows`` off (side "off", main's default) against on.

Both engines come from the minimal planner for the same structure, MH-1 hybrid float32 NVT, and differ
only in ``RewriteConfig(gated_rows="on")``. The record changes only modules of its own model copy, so the
two share no process-wide state. After the warm-up their timed blocks alternate (``fastmace.bench.ab.run_ab``,
12 blocks of 300 steps, ABAB or ABBA), and the result is the ratio of medians on/off. Writes ``<output>.json``.
Run on an otherwise idle GPU; ``ab_step_time.compile_counters`` must show no compilation while timed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(fastmace.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"])
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--pattern", choices=("abab", "abba"), default="abab")
    p.add_argument("--order", nargs=2, default=["off", "on"], choices=("off", "on"), help="Build order")
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from torch._dynamo.utils import counters

    from fastmace.bench import ab, step_time
    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    meta = dict(tool="opt-T02a ab_gated_rows", created_utc=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, checkpoint_sha256=sha,
                protocol=protocol.as_dict(), pattern=args.pattern, order=args.order,
                load=list(os.getloadavg()), structures={})
    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        sides, setup, lines = {}, {}, {}
        for side in args.order:
            started = time.perf_counter()
            rewrites = RewriteConfig(gated_rows="on") if side == "on" else "auto"
            field = ForceField(loaded, base, precision="float32", rewrites=rewrites)
            integrator = IntegratorConfig(temperature_K=args.temperature_K)
            plan = planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
            engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K)
            sides[side] = ab_step_time.FastMACESide(engine, torch)
            setup[side] = time.perf_counter() - started
            lines[side] = [line.strip() for line in plan.describe().splitlines() if "gated_rows" in line]
        for side in sides:  # warm up outside the compile-counter window (new edge sizes compile here)
            step_time.warm_up(sides[side], protocol.warmup)
        counted = ab_step_time.compile_counters(counters)
        result = ab.run_ab(sides, dataclasses.replace(protocol, warmup=0), pattern=args.pattern)
        changes, recompiled = ab_step_time.compile_changes(counted, ab_step_time.compile_counters(counters))
        medians = {side: result.statistics[side].median_ms for side in sides}
        blocks = {side: result.statistics[side].block_ms for side in sides}
        entry = dict(natoms=len(base), ab=result.as_dict(), ratio_on_over_off=medians["on"] / medians["off"],
                     pair_ratios=[on / off for on, off in zip(blocks["on"], blocks["off"])],
                     setup_s=setup, plan_lines=lines,
                     compile_during_timing=dict(changes=changes, recompiled=recompiled),
                     observables={side: sides[side].observables() for side in sides},
                     load_after=list(os.getloadavg()),
                     other_gpu_processes=[pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()])
        entry["valid"] = not recompiled and not entry["other_gpu_processes"]
        meta["structures"][name] = entry
        print(name, " ".join(f"{s} {medians[s]:.4f} ms" for s in sides), f"on/off {entry['ratio_on_over_off']:.4f}",
              "valid" if entry["valid"] else "INVALID", flush=True)
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        sides = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
