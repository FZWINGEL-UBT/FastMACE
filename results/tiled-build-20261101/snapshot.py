"""What the force field's construction holds on the device at its peak (tiled path, as tiled_capacity.py builds it
with --tile-list): PyTorch's memory history during the build, replayed to the peak, and the allocations alive
there, largest first, each with its innermost FastMACE or MACE frames.

Run as ``run.sh NAME results/tiled-build-20261101/snapshot.py --structure NbCl5 --repeat 45x45x135``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
import tiled_capacity  # noqa: E402


def where(frames):
    """The innermost frames in fastmace or mace, as ``file:line function``."""
    keep = [f"{f['filename'].split('site-packages/')[-1].split('src-build')[-1]}:{f['line']} {f['name']}"
            for f in frames if "fastmace" in f["filename"] or "/mace/" in f["filename"]]
    return keep[:4]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", required=True)
    p.add_argument("--receivers", type=int, default=4096)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    with contextlib.ExitStack() as stack:
        stack.enter_context(tiled_capacity.no_global_list(True))
        user = fm.ForceField(fm.load(args.model), atoms, precision="float32")
        planner = minimal.default_planner(user)
        plan = planner.plan_evaluation(user, atoms)
        eager = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                       dynamic_edges=False))
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=200_000, stacks="python")
        field = planner.build_evaluation(eager, user, atoms, call_rows=args.receivers)
        torch.cuda.synchronize()
        snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        peak = torch.cuda.max_memory_allocated() - start
        after = torch.cuda.memory_allocated() - start

    live, total, best, best_live = {}, 0, 0, {}
    for event in snapshot["device_traces"][0]:
        action = event["action"]
        if action == "alloc":
            live[event["addr"]] = event
            total += event["size"]
            if total > best:
                best, best_live = total, dict(live)
        elif action == "free_completed" and event["addr"] in live:
            total -= live.pop(event["addr"])["size"]
    n = len(atoms)
    rows = sorted(best_live.values(), key=lambda e: -e["size"])
    top = [dict(bytes=e["size"], bytes_per_atom=round(e["size"] / n, 1), where=where(e["frames"])) for e in rows[:25]]
    record = dict(structure=args.structure, atoms=n, source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"),
                  device=torch.cuda.get_device_name(), host=os.uname().nodename,
                  allocated_before_build=start, build_peak_above=peak, kept_after_build=after,
                  traced_peak=best, traced_peak_per_atom=round(best / n, 1), live_at_peak=top,
                  small_at_peak=sum(e["size"] for e in rows[25:]), applied=list(getattr(field, "applied", ())))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"snapshot-{args.structure}-{n}.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"peak {peak / 2**30:.2f} GiB above start, traced {best / 2**30:.2f} GiB, kept {after / 2**30:.2f} GiB")
    for row in top:
        print(f"{row['bytes'] / 2**30:7.3f} GiB {row['bytes_per_atom']:8.1f} B/atom  {' <- '.join(row['where'][:3])}")


if __name__ == "__main__":
    main()
