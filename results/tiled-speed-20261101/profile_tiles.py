"""Where a tiled call's time goes (the owner's speed phase): one eager or compiled tiled call, stage by stage.

The tiled force field of the measured limits: R = 4,096, tile lists, the state on the host and aliased, the species
index; ``--mode eager`` or ``compiled`` (the default compiled evaluation plan's traceable model). After warm-up:

1. **A timed call.** The tiled path's stages are wrapped from outside (no change to the library); each stage
   boundary synchronises the device, so a stage owns the device time of what it launched, exclusive of the stages
   nested in it. Stages: the tile graph (with the per-tile neighbour list), the host state's reads, writes,
   window and closing, the sweep bodies (interaction 0, interaction 1), the padded fill, body and scatter, and
   the rest of the call.
2. **A profiled call** (torch.profiler, CPU and CUDA, no synchronisation): the device time by kernel, grouped, and
   the device's busy share of the call's wall time.

Run as ``run.sh NAME results/tiled-profile-20261101/profile_tiles.py --repeat 16x16x48 --mode eager``.
"""

from __future__ import annotations

import argparse
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

GROUPS = (  # kernel-name substring -> group, first match wins
    ("memcpy", "copies"), ("Memcpy", "copies"), ("memset", "copies"), ("Memset", "copies"),
    ("segmented", "convolution"), ("conv", "convolution"), ("tensor_product", "convolution"),
    ("uniform_1d", "cuequivariance"), ("kernelcatcher", "cuequivariance"), ("cue", "cuequivariance"),
    ("gemm", "gemm"), ("Gemm", "gemm"), ("cutlass", "gemm"), ("sm80_", "gemm"), ("sm90_", "gemm"),
    ("ampere", "gemm"), ("index", "index/scatter"), ("scatter", "index/scatter"), ("gather", "index/scatter"),
    ("sort", "sort/unique"), ("unique", "sort/unique"), ("radix", "sort/unique"), ("triton", "triton"),
    ("elementwise", "elementwise"), ("reduce", "reduce"),
)


class Stages:
    """Exclusive wall time per stage, with the device synchronised at every boundary."""

    def __init__(self, torch):
        self.torch, self.totals, self.counts, self.stack = torch, collections.Counter(), collections.Counter(), []

    def wrap(self, owner, name, label):
        original = getattr(owner, name)
        stages = self

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            stages.enter(label)
            try:
                return original(*args, **kwargs)
            finally:
                stages.leave(label)

        setattr(owner, name, wrapped)

    def enter(self, label):
        self.torch.cuda.synchronize()
        now = time.perf_counter()
        if self.stack:
            parent, started = self.stack[-1]
            self.totals[parent] += now - started
        self.stack.append((label, now))
        self.counts[label] += 1

    def leave(self, label):
        self.torch.cuda.synchronize()
        now = time.perf_counter()
        _, started = self.stack.pop()
        self.totals[label] += now - started
        if self.stack:
            self.stack[-1] = (self.stack[-1][0], now)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", required=True)
    p.add_argument("--receivers", type=int, default=4096)
    p.add_argument("--mode", choices=("eager", "compiled"), default="eager")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

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
        plain = call()
        stages = Stages(torch)
        for owner, name, label in (
                (TileNeighbours, "edges", "tile list (per-tile neighbour list)"),
                (TiledForceField, "_tile_graph", "tile graph (rest)"),
                (tiled_module._State, "rows", "host state: read rows"),
                (tiled_module._State, "put", "host state: put rows"),
                (tiled_module._State, "add", "host state: add rows / window"),
                (tiled_module._State, "touch", "host state: touch"),
                (tiled_module._State, "plan", "host state: plan"),
                (tiled_module._State, "close", "host state: close"),
                (TiledForceField, "_first", "interaction 0 (sweeps 1, 3)"),
                (TiledForceField, "_second", "interaction 1 (sweep 2)"),
                (TiledForceField, "_fill", "padded: fill"),
                (TiledForceField, "_run_slot", "padded: body (rest)"),
                (TiledForceField, "_scatter", "padded: scatter (rest)"),
                (TiledForceField, "_eager_tile", "eager tile (rest: scatter)"),
                (TiledForceField, "__call__", "call (rest)")):
            stages.wrap(owner, name, label)
        timed = call()
        totals = {label: round(seconds, 3) for label, seconds in stages.totals.most_common()}
        # The profiled call runs on the unwrapped methods: the wrappers synchronise.
        for owner, name in ((TileNeighbours, "edges"), (TiledForceField, "_tile_graph"), (tiled_module._State, "rows"),
                            (tiled_module._State, "put"), (tiled_module._State, "add"),
                            (tiled_module._State, "touch"), (tiled_module._State, "plan"),
                            (tiled_module._State, "close"), (TiledForceField, "_first"),
                            (TiledForceField, "_second"), (TiledForceField, "_fill"), (TiledForceField, "_run_slot"),
                            (TiledForceField, "_scatter"), (TiledForceField, "_eager_tile"),
                            (TiledForceField, "__call__")):
            setattr(owner, name, getattr(owner, name).__wrapped__)
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            profiled = call()
    from torch.autograd import DeviceType

    kernels, groups = collections.Counter(), collections.Counter()
    for event in prof.events():  # device events only (kernels, copies, sets): operators would count them again
        if event.device_type == DeviceType.CUDA:
            kernels[event.name] += getattr(event, "device_time_total", None) or event.cuda_time_total
    kernels = collections.Counter({name: microseconds / 1e6 for name, microseconds in kernels.items()})
    for key, seconds in kernels.items():
        group = next((g for pattern, g in GROUPS if pattern in key), "other")
        groups[group] += seconds
    device_total = sum(kernels.values())
    record = dict(structure=args.structure, atoms=n, mode=args.mode, receivers=args.receivers,
                  device=torch.cuda.get_device_name(), host=os.uname().nodename,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), warm=warm, plain_call=plain,
                  timed_call=timed, stages=totals, stage_counts=dict(stages.counts), profiled_call=profiled,
                  device_seconds=device_total, device_busy_share=device_total / profiled,
                  kernel_groups={g: round(s, 3) for g, s in groups.most_common()},
                  top_kernels=[(k[:120], round(s, 3)) for k, s in kernels.most_common(25)],
                  tiles=tiled.last.get("tiles"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"profile-{args.mode}-{args.structure}-{n}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "top_kernels"}))
    for k, s in record["top_kernels"][:15]:
        print(f"{s:9.3f} s  {k}")


if __name__ == "__main__":
    main()
