"""What a tiled call holds on the device at its peak, call by call: PyTorch's memory history over ``--calls`` calls,
each call's trace replayed to its own peak, and the allocations alive there, largest first, with their innermost
FastMACE frames; and the call's largest single allocations wherever they fall (the per-atom transients of the
whole system, which decide the peak at the largest sizes). The force field as tiled_capacity.py builds it (tile
lists, the state on the host and aliased, the species index); eager tiles.

Run as ``run.sh NAME results/tiled-call-memory-20261101/snapshot_calls.py --repeat 35x35x105`` (the history is a
ring buffer of ``--entries`` events: a call must fit in it).
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
    keep = [f"{f['filename'].split('fastmace/')[-1]}:{f['line']} {f['name']}" for f in frames
            if "/fastmace/" in f["filename"] or "tools/bench" in f["filename"]]
    return keep[:4]


def peak_of(trace, start):
    """The live allocations at the trace's peak, from ``start`` (the events before belong to earlier calls)."""
    live, total, best, best_live = {}, 0, 0, {}
    for event in trace[start:]:
        if event["action"] == "alloc":
            live[event["addr"]] = event
            total += event["size"]
            if total > best:
                best, best_live = total, dict(live)
        elif event["action"] == "free_completed" and event["addr"] in live:
            total -= live.pop(event["addr"])["size"]
    return best, best_live


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", required=True)
    p.add_argument("--calls", type=int, default=2)
    p.add_argument("--entries", type=int, default=4_000_000)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    calls = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(tiled_capacity.no_global_list(True))
        user = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
        planner = minimal.default_planner(user)
        plan = planner.plan_evaluation(user, atoms)
        eager = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                       dynamic_edges=False))
        field = planner.build_evaluation(eager, user, atoms, call_rows=4096, release_node_attrs=True)
        tiled = TiledForceField(field, TileConfig(4096, mode="eager", neighbours="tiles", state="host",
                                                  alias_state=True, species="index"))
        positions = field._prepare_one_off(atoms)
        out = None
        for index in range(args.calls):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.memory._record_memory_history(max_entries=args.entries, stacks="python")
            out = tiled(positions)  # the previous call's outputs are alive meanwhile, as in a caller's loop
            torch.cuda.synchronize()
            snapshot = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)
            trace = snapshot["device_traces"][0]
            traced, live = peak_of(trace, 0)
            rows = sorted(live.values(), key=lambda e: -e["size"])
            largest, seen = [], set()
            for e in sorted((e for e in trace if e["action"] == "alloc"), key=lambda e: -e["size"]):
                site = tuple(where(e["frames"])[:2])
                if site not in seen:
                    seen.add(site)
                    largest.append(dict(bytes=e["size"], bytes_per_atom=round(e["size"] / n, 1), where=list(site)))
                if len(largest) == 20:
                    break
            calls.append(dict(call=index + 1, allocated_before=before,
                              peak_above=torch.cuda.max_memory_allocated() - before, traced_peak=traced,
                              live_at_peak=[dict(bytes=e["size"], bytes_per_atom=round(e["size"] / n, 1),
                                                 where=where(e["frames"])) for e in rows[:30]],
                              small=sum(e["size"] for e in rows[30:]), events=len(trace),
                              largest_allocations=largest))
        del out
    record = dict(structure=args.structure, atoms=n, host=os.uname().nodename,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), device=torch.cuda.get_device_name(),
                  calls=calls)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"snapshot-calls-{args.structure}-{n}.json").write_text(json.dumps(record, indent=1))
    for c in calls:
        print(f"call {c['call']}: before {c['allocated_before'] / 2**30:.2f} GiB, peak above "
              f"{c['peak_above'] / 2**30:.2f} GiB, traced {c['traced_peak'] / 2**30:.2f} GiB")
        for row in c["live_at_peak"][:12]:
            print(f"  {row['bytes'] / 2**30:7.3f} GiB {row['bytes_per_atom']:7.1f} B/atom  "
                  f"{' <- '.join(row['where'][:3])}")
        print(f"  largest single allocations of call {c['call']} ({c['events']} events):")
        for row in c["largest_allocations"][:14]:
            print(f"  {row['bytes'] / 2**30:7.3f} GiB {row['bytes_per_atom']:7.1f} B/atom  {' <- '.join(row['where'])}")


if __name__ == "__main__":
    main()
