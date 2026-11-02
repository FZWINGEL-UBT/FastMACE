"""The radial MLP's share of a compiled tiled call (generic models such as MACE-OMAT-0): a bound for radial tables.

The tiled force field of profile_tiles.py (R = ``--receivers`` without the budget, tile lists, the state on the host
and aliased, the species index), compiled. After two warm-up calls, one call under torch.profiler with the input
shapes recorded:

- every matrix product (``aten::mm``, ``addmm``, ``bmm``, ``linear``) with its shapes and device time. The radial
  MLP's run over a tile's edges (M = the tile's edge count, about 74 per receiver), the node-wise linears over its
  rows (R + H); the shapes tell them apart;
- every device kernel whose name contains ``silu`` (the radial MLP's activation; MACE's other nonlinearities are
  gates on node features) with its device time.

Run as ``run.sh NAME results/tiled-omat-20261102/radial_profile.py --repeat 58x58x58``.
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

PRODUCTS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::linear", "aten::matmul")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="SrTiO3")
    p.add_argument("--model", default="omat0-medium")
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
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    with contextlib.ExitStack() as stack:
        stack.enter_context(tiled_capacity.no_global_list(True))
        user = fm.ForceField(fm.load(args.model), atoms, precision="float32")
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
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            profiled = call()
    products = collections.defaultdict(lambda: [0, 0.0])
    for entry in prof.key_averages(group_by_input_shape=True):
        if entry.key in PRODUCTS:
            time_us = getattr(entry, "device_time_total", None) or entry.cuda_time_total
            products[(entry.key, str(entry.input_shapes))][0] += entry.count
            products[(entry.key, str(entry.input_shapes))][1] += time_us / 1e6
    silu, kernels = collections.Counter(), collections.Counter()
    for event in prof.events():
        if event.device_type == DeviceType.CUDA:
            seconds = (getattr(event, "device_time_total", None) or event.cuda_time_total) / 1e6
            kernels[event.name[:140]] += seconds
            if "silu" in event.name:
                silu[event.name[:140]] += seconds
    record = dict(structure=args.structure, model=args.model, atoms=n, receivers=args.receivers,
                  device=torch.cuda.get_device_name(), host=os.uname().nodename,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), warm=warm, profiled_call=profiled,
                  tiles=tiled.last.get("tiles"), edges=tiled.last.get("edges"), max_edges=tiled.last.get("max_edges"),
                  slots={str(k): list(v) for k, v in tiled.last.get("slots", {}).items()},
                  device_seconds=sum(kernels.values()),
                  products=sorted(([k[0], k[1], c, round(s, 4)] for k, (c, s) in products.items()),
                                  key=lambda row: -row[3]),
                  silu_kernels={k: round(v, 4) for k, v in silu.most_common()},
                  silu_seconds=sum(silu.values()),
                  top_kernels=[(k, round(v, 4)) for k, v in kernels.most_common(30)])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"radial-{args.model}-{args.structure}-{n}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("atoms", "profiled_call", "device_seconds", "silu_seconds", "tiles",
                                              "edges", "max_edges", "slots")}))
    for row in record["products"][:25]:
        print(f"{row[3]:8.3f} s  {row[2]:6d}x  {row[0]}  {row[1][:160]}")


if __name__ == "__main__":
    main()
