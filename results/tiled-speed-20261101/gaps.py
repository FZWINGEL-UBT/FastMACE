"""Where the device idles in a tiled call: the gaps between device activities, by length and by the host's stage.

The tiled force field of profile_tiles.py (R = 4,096, tile lists with the cache, the state on the host and aliased,
the species index, the overlap at its default), ``--mode eager`` or ``compiled``. After warm-up, one call under
torch.profiler (CPU and CUDA). The library's stages are marked with ``record_function`` (no synchronisation).

- The device's activities (kernels, copies, sets) are merged into busy intervals; the gaps between them are
  bucketed by length (count and total).
- Each gap is charged to the innermost marked stage the host was in at the gap's start, i.e. when the device ran
  dry. Short gaps inside the sweep bodies are launch and Python overhead (a CUDA graph would remove them); long
  gaps under the tile list or the host state are host work.

Run as ``run.sh NAME results/tiled-speed-20261101/gaps.py --repeat 16x16x48 --mode compiled``.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import contextlib
import dataclasses
import functools
import json
import os
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
import tiled_capacity  # noqa: E402

BUCKETS = (10, 50, 200, 1000, 10_000, float("inf"))  # microseconds, upper bounds
PREFIX = "stage: "


def mark(torch, owner, name, label):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(PREFIX + label):
            return original(*args, **kwargs)

    setattr(owner, name, wrapped)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", required=True)
    p.add_argument("--receivers", type=int, default=4096)
    p.add_argument("--mode", choices=("eager", "compiled"), default="compiled")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    import fastmace as fm
    from fastmace.model import tiled as tiled_module
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.neighbors.tile_list import TileNeighbours
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    with contextlib.ExitStack() as stack:
        stack.enter_context(tiled_capacity.no_global_list(True))
        user = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
        planner = minimal.default_planner(user)
        plan = planner.plan_evaluation(user, atoms)
        if args.mode == "eager":
            plan = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                          dynamic_edges=False))
        field = planner.build_evaluation(plan, user, atoms, call_rows=args.receivers, release_node_attrs=True)
        tiled = TiledForceField(field, TileConfig(args.receivers, mode=args.mode, neighbours="tiles", state="host",
                                                  alias_state=True, species="index"))
        positions = field._prepare_one_off(atoms)

        def call():
            torch.cuda.synchronize()
            started = time.perf_counter()
            tiled(positions)
            torch.cuda.synchronize()
            return time.perf_counter() - started

        warm = [call(), call()]
        for owner, name, label in (
                (TileNeighbours, "edges", "tile list"),
                (TiledForceField, "_tile_graph", "tile graph"),
                (tiled_module._State, "rows", "host state: read"),
                (tiled_module._State, "put", "host state: put"),
                (tiled_module._State, "add", "host state: add"),
                (tiled_module._State, "flush", "host state: flush"),
                (tiled_module._State, "touch", "host state: touch"),
                (tiled_module._State, "plan", "host state: plan"),
                (tiled_module._State, "close", "host state: close"),
                (TiledForceField, "_first", "interaction 0"),
                (TiledForceField, "_second", "interaction 1"),
                (TiledForceField, "_fill", "padded: fill"),
                (TiledForceField, "_run_slot", "padded: body"),
                (TiledForceField, "_scatter", "padded: scatter"),
                (TiledForceField, "_eager_tile", "eager tile"),
                (TiledForceField, "__call__", "call")):
            if hasattr(owner, name):
                mark(torch, owner, name, label)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            profiled = call()

    device, stages = [], []
    for event in prof.events():
        if event.name.startswith(PREFIX) and event.device_type == DeviceType.CUDA:
            continue  # the marks' device-side annotations span whole stages: not device work
        if event.device_type == DeviceType.CUDA:
            device.append((event.time_range.start, event.time_range.end))
        elif event.name.startswith(PREFIX):
            stages.append((event.time_range.start, event.time_range.end, event.name[len(PREFIX):]))
    device.sort()
    busy = []  # merged busy intervals
    for start, end in device:
        if busy and start <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], end)
        else:
            busy.append([start, end])
    stages.sort()
    starts = [s[0] for s in stages]

    def stage_at(t):  # innermost marked stage covering t: the latest-starting one that has not ended
        for i in range(bisect.bisect_right(starts, t) - 1, -1, -1):
            start, end, label = stages[i]
            if end >= t:
                return label
        return "(outside the stages)"

    buckets = collections.OrderedDict((f"<= {b} us" if b != float("inf") else "> 10 ms", [0, 0.0]) for b in BUCKETS)
    by_stage = collections.defaultdict(lambda: collections.OrderedDict((k, [0, 0.0]) for k in buckets))
    for (_, end), (start, _) in zip(busy, busy[1:]):
        gap = start - end
        key = list(buckets)[next(i for i, b in enumerate(BUCKETS) if gap <= b)]
        label = stage_at(end)
        for table in (buckets, by_stage[label]):
            table[key][0] += 1
            table[key][1] += gap / 1e6
    busy_seconds = sum(e - s for s, e in busy) / 1e6
    span = (busy[-1][1] - busy[0][0]) / 1e6
    record = dict(structure=args.structure, atoms=n, mode=args.mode, receivers=args.receivers,
                  device=torch.cuda.get_device_name(), host=os.uname().nodename,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), warm=warm, profiled_call=profiled,
                  device_activities=len(device), busy_intervals=len(busy), busy_seconds=busy_seconds,
                  device_span_seconds=span, idle_seconds=span - busy_seconds,
                  gaps={k: dict(count=c, seconds=round(s, 4)) for k, (c, s) in buckets.items()},
                  gaps_by_stage={label: {k: dict(count=c, seconds=round(s, 4)) for k, (c, s) in table.items() if c}
                                 for label, table in sorted(by_stage.items(),
                                                            key=lambda kv: -sum(s for _, s in kv[1].values()))},
                  tiles=tiled.last.get("tiles"))
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np  # the raw intervals, for analyses without the GPU

    np.savez_compressed(args.output / f"intervals-{args.mode}-{args.structure}-{n}.npz",
                        device=np.array(device, dtype=np.float64).reshape(-1, 2),
                        stages=np.array([(s, e) for s, e, _ in stages], dtype=np.float64).reshape(-1, 2),
                        stage_labels=np.array([label for _, _, label in stages]))
    (args.output / f"gaps-{args.mode}-{args.structure}-{n}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "gaps_by_stage"}))
    for label, table in record["gaps_by_stage"].items():
        total = sum(v["seconds"] for v in table.values())
        print(f"{total:8.3f} s  {label:22s} " + "  ".join(f"{k}: {v['count']} / {v['seconds']:.3f} s"
                                                         for k, v in table.items()))


if __name__ == "__main__":
    main()
