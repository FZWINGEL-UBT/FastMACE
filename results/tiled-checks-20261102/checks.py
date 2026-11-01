"""Two checks on a compiled tiled call (conductor, 2 November): which gated-products input-gradient kernel runs, and
the tile graph's time per sweep.

The tiled force field of profile_tiles.py: the default compiled plan, tile lists with the cache, the state on the host
and aliased, the species index, R = ``--receivers`` without the budget. After two warm-up calls:

1. **A timed call.** ``_tile_graph`` is wrapped with device synchronisation. Its calls come in four rounds of one per
   tile: the slots' planning (``_slot_keys``), then sweeps 1, 2 and 3. Each round's time and call count are reported,
   and so is the whole call.
2. **A profiled call** (torch.profiler, CUDA, without the wrapper): every kernel whose name contains "input_grads",
   "unpadded" or "gated", with its launches and device time.

Run as ``run.sh NAME results/tiled-checks-20261102/checks.py --repeat 16x16x48 --receivers 16384``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
import tiled_capacity  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", required=True)
    p.add_argument("--receivers", type=int, default=16384)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    import fastmace as fm
    from fastmace.kernels.triton import gated_products
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    with contextlib.ExitStack() as stack:
        stack.enter_context(tiled_capacity.no_global_list(True))
        user = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
        planner = minimal.default_planner(user)
        plan = planner.plan_evaluation(user, atoms)
        field = planner.build_evaluation(plan, user, atoms, call_rows=args.receivers, release_node_attrs=True)
        tiled = TiledForceField(field, TileConfig(args.receivers, mode="compiled", neighbours="tiles", state="host",
                                                  alias_state=True, species="index", budget_receivers=()))
        positions = field._prepare_one_off(atoms)

        def call():
            torch.cuda.synchronize()
            started = time.perf_counter()
            tiled(positions)
            torch.cuda.synchronize()
            return time.perf_counter() - started

        warm = [call(), call()]
        original = TiledForceField._tile_graph
        times = []

        def timed(self, *a, **k):
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = original(self, *a, **k)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            return out

        TiledForceField._tile_graph = timed
        timed_call = call()
        TiledForceField._tile_graph = original
        tiles = tiled.last["tiles"]
        names = ("slot planning", "sweep 1", "sweep 2", "sweep 3")
        rounds = {name: dict(calls=len(times[i * tiles:(i + 1) * tiles]),
                             seconds=round(sum(times[i * tiles:(i + 1) * tiles]), 4))
                  for i, name in enumerate(names)}
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            profiled = call()
    kernels = collections.defaultdict(lambda: [0, 0.0])
    for event in prof.events():
        if event.device_type == DeviceType.CUDA and any(s in event.name for s in ("input_grads", "unpadded", "gated")):
            kernels[event.name][0] += 1
            kernels[event.name][1] += (getattr(event, "device_time_total", None) or event.cuda_time_total) / 1e6
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    record = dict(structure=args.structure, atoms=n, receivers=args.receivers, device=torch.cuda.get_device_name(),
                  sms=sms, host=os.uname().nodename, source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"),
                  unpadded_from_rows=gated_products.UNPADDED_ATOMS_PER_SM * sms,
                  slots={str(k): list(v) for k, v in tiled.last.get("slots", {}).items()}, tiles=tiles,
                  warm=warm, timed_call=timed_call, tile_graph_rounds=rounds,
                  tile_graph_calls=len(times), profiled_call=profiled,
                  kernels={k[:160]: dict(launches=c, seconds=round(s, 4)) for k, (c, s) in
                           sorted(kernels.items(), key=lambda kv: -kv[1][1])})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"checks-{args.structure}-{n}-R{args.receivers}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
