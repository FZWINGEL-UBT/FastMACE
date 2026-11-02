"""Two tiles at once: stages 0 and 1 of ``plans/tiled-concurrent-tiles.md`` (measurement only, no library change).

Stage 0: a kernel-class profile of one compiled tiled call (torch.profiler): each class's share of the kernel
time and how busy the device is (the union of kernel intervals over the call's wall time).

Stage 1: the co-run probe. A further tiled call records copies of the padded slots of two tiles per sweep; the
probe then replays pairs of tile-sweeps (each lane on its own slot copy, reading the call's device state):

- ``serial``: both lanes back to back on one stream (the baseline);
- ``streams``: each lane on its own stream;
- ``split``: each lane on its own stream of a green context holding half the SMs (disjoint halves);
- with ``--tf32x3``, a second force field with ``tf32x3_gemms="on"`` gives lane B (its GEMMs as 3xTF32 on the
  tensor cores) beside lane A in IEEE FP32: ``serial`` and ``streams`` of that pair.

Each mode runs ``--reps`` pairs per block, in ``--blocks`` blocks with the modes' order rotated, in one process;
the result is the median time per pair and its ratio to ``serial``.

State on the device (no host state, no aliasing), so the sweeps read no host rows. Writes
``<output>/corun-<structure>-<atoms>-R<receivers>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiled_capacity  # noqa: E402

CLASSES = (  # first match wins
    ("cueq_kernel", re.compile(r"^kernel$")),  # a JIT kernel named only "kernel" (cuEquivariance's)
    ("conv", re.compile(r"oeq|openequivariance|conv|tensor_product|streaming|segmented|cuequivariance|uniform_1d|"
                        r"symmetric|tp_fwd|tp_bwd|fused_tp", re.I)),
    ("gemm", re.compile(r"gemm|sgemm|cutlass|cublas|xmma|ampere_|sm80_|sm89_|sm90_|sm100_|sm120_|tem_fused|"
                        r"triton_tem|_mm_|tf32x3|matmul", re.I)),
    ("index", re.compile(r"index|scatter|gather|indexselect|embedding|one_hot|take|put_", re.I)),
    ("copy", re.compile(r"memcpy|memset|copy|cat_|fill", re.I)),
    ("pointwise", re.compile(r"triton_poi|triton_per|triton_red|elementwise|reduce|vectorized|unrolled|"
                             r"sum_|norm|where|silu|sigmoid|mul|add", re.I)),
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="SrTiO3")
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--receivers", type=int, default=8192)
    p.add_argument("--reps", type=int, default=4, help="pairs per block")
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--tf32x3", action="store_true", help="also lane B with tf32x3_gemms='on' (grouped_launch off: both reroute the block linears)")
    p.add_argument("--no-split", action="store_true", help="skip the green-context mode")
    p.add_argument("--split-only", action="store_true",
                   help="only serial against split, without expandable segments (their mappings fail on green-"
                        "context streams: 'device not ready', 2 November 2026)")
    p.add_argument("--no-expandable", action="store_true",
                   help="PyTorch's default allocator instead of expandable segments (implied by --split-only); a "
                        "PYTORCH_CUDA_ALLOC_CONF set by the caller (e.g. backend:cudaMallocAsync) is kept either way")
    p.add_argument("--tag", default="")
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(argv)


def classify(name):
    for label, pattern in CLASSES:
        if pattern.search(name):
            return label
    return "other"


def profile_call(tiled, positions, host_ops=0):
    """Stage 0: one compiled call under torch.profiler; kernel time by class and the device's busy fraction. With
    ``host_ops``, returns ``(summary, output)`` and the summary lists the host operations with the most self time."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        started = time.perf_counter()
        output = tiled(positions)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    intervals = sorted((e.time_range.start, e.time_range.end) for e in kernels)
    busy, end = 0.0, None
    for start, stop in intervals:  # the union of kernel intervals (µs)
        if end is None or start > end:
            busy += stop - start
            end = stop
        elif stop > end:
            busy += stop - end
            end = stop
    span = (intervals[-1][1] - intervals[0][0]) if intervals else 0.0
    classes, top = {}, {}
    for e in kernels:
        us = e.time_range.end - e.time_range.start
        label = classify(e.name)
        classes[label] = classes.get(label, 0.0) + us
        top[e.name] = top.get(e.name, 0.0) + us
    total = sum(classes.values()) or 1.0
    summary = dict(wall_s=wall, kernel_s=total / 1e6, busy_s=busy / 1e6, span_s=span / 1e6,
                busy_fraction_of_span=busy / span if span else None,
                kernel_sum_over_busy=total / busy if busy else None,
                classes={k: dict(seconds=v / 1e6, share=v / total) for k, v in sorted(classes.items(),
                                                                                     key=lambda kv: -kv[1])},
                top=[dict(name=k[:160], seconds=v / 1e6, share=v / total, cls=classify(k))
                     for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:30]])
    if not host_ops:
        return summary
    averages = prof.key_averages()
    summary["host_ops"] = [dict(name=a.key[:120], self_cpu_s=a.self_cpu_time_total / 1e6, calls=a.count)
                           for a in sorted(averages, key=lambda a: -a.self_cpu_time_total)[:host_ops]]
    return summary, output


def capture_slots(tiled, positions, per_sweep=2, skip=1):
    """Run one call and keep copies of the padded slots of ``per_sweep`` tiles of each sweep (after ``skip``)."""
    import torch

    seen, kept = {1: 0, 2: 0, 3: 0}, []
    run_slot = tiled._run_slot

    def recording(sweep, slot, g):
        k = seen[sweep]
        seen[sweep] += 1
        if skip <= k < skip + per_sweep:
            copy = {name: slot[name].clone() for name in ("nodes", "node_mask", "recv_mask", "ei", "shifts",
                                                          "local", "edge_mask", "count")}
            copy.update(sizes=slot["sizes"], tables=[t.clone() for t in slot["tables"]], out={}, graphs={})
            kept.append((sweep, copy))
        run_slot(sweep, slot, g)

    tiled._run_slot = recording
    try:
        tiled(positions)
    finally:
        del tiled._run_slot
    torch.cuda.synchronize()
    return kept, tiled._g


class Lane:
    """One tile-sweep on one slot copy of one tiled force field."""

    def __init__(self, tiled, g, sweep, slot):
        self.tiled, self.g, self.sweep, self.slot = tiled, g, sweep, slot
        self.label = f"sweep{sweep}/R{slot['sizes'][0]}/E{int(slot['count'])}"

    def launch(self):
        tiled = self.tiled
        for module in tiled.field._streaming:  # the streaming kernels read this slot's edge count (at launch)
            module.register_buffer("streaming_count", self.slot["count"], persistent=False)
        tiled._bind_tables(self.slot["tables"])
        tiled._body(self.sweep, self.slot, self.g)


def green_streams():
    """Two streams on disjoint halves of the device's SMs (two green contexts from one split), or None."""
    import torch
    from torch.cuda import green_contexts as gc

    drv = gc._drv
    check = gc._check_cuda_bindings
    device_id = torch.cuda.current_device()
    device = check(drv.cuDeviceGet(device_id))
    sm = check(drv.cuDeviceGetDevResource(device, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    count = sm.sm.smCount
    major = torch.cuda.get_device_capability(device_id)[0]
    step = 8 if major >= 9 else 2  # the split granularity: 2 SMs on sm_8x, 8 from sm_90 on
    half = (count // 2) // step * step
    groups, n_groups, _ = check(drv.cuDevSmResourceSplitByCount(2, sm, 0, half))
    if n_groups != 2:
        return None, dict(sms=count, half=half, groups=n_groups)
    streams, contexts = [], []
    for group in list(groups)[:2]:
        desc = check(drv.cuDevResourceGenerateDesc([group], 1))
        green = check(drv.cuGreenCtxCreate(desc, device, drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        context = check(drv.cuCtxFromGreenCtx(green))
        ctx = gc.GreenContext.__new__(gc.GreenContext)
        ctx._init_from_cuda_objects(device_id, green, context)
        contexts.append(ctx)
        streams.append(ctx.Stream())
    return (streams, contexts), dict(sms=count, half=half, groups=n_groups)


def run_pair(a, b, mode, reps, streams):
    import torch

    current = torch.cuda.current_stream()
    torch.cuda.synchronize()
    started = time.perf_counter()
    if mode == "serial":
        for _ in range(reps):
            a.launch()
            b.launch()
    else:
        s1, s2 = streams[mode]
        s1.wait_stream(current)
        s2.wait_stream(current)
        for _ in range(reps):
            with torch.cuda.stream(s1):
                a.launch()
            with torch.cuda.stream(s2):
                b.launch()
        current.wait_stream(s1)
        current.wait_stream(s2)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / reps


def run_alone(lane, reps):
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        lane.launch()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / reps


def build(args, atoms, rewrites=None):
    import dataclasses  # noqa: F401

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    kwargs = dict(rewrites=rewrites) if rewrites is not None else {}
    user = fm.ForceField(fm.load(args.model), atoms, precision="float32", **kwargs)
    planner = minimal.default_planner(user)
    plan = planner.plan_evaluation(user, atoms)
    field = planner.build_evaluation(plan, user, atoms, call_rows=args.receivers, release_node_attrs=True)
    tiled = TiledForceField(field, TileConfig(args.receivers, None, mode="compiled", neighbours="tiles",
                                              budget_receivers=(), state="device", wavefront="off",
                                              species="index"))
    return user, field, tiled


def main(argv=None):
    args = parse_args(argv)
    if not (args.split_only or args.no_expandable):
        from fastmace.model.compile import configure_allocator

        configure_allocator()
    import torch

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    record = dict(structure=args.structure, repeat=list(repeat), atoms=n, receivers=args.receivers,
                  model=args.model, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  host=os.uname().nodename, source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"),
                  reps=args.reps, blocks=args.blocks,
                  expandable_segments=not (args.split_only or args.no_expandable),
                  alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"corun-{args.structure}-{n}-R{args.receivers}{'-split' if args.split_only else ''}{args.tag}.json"

    def save():
        path.write_text(json.dumps(record, indent=1, default=str))

    with tiled_capacity.no_global_list(True):
        fields = {}
        user, field, tiled = build(args, atoms)
        positions = field._prepare_one_off(atoms)
        fields["A"] = (field, tiled, positions)
        if args.tf32x3:
            from fastmace.config import RewriteConfig

            user_b, field_b, tiled_b = build(args, atoms, RewriteConfig(tf32x3_gemms="on", grouped_launch="off"))
            fields["B"] = (field_b, tiled_b, field_b._prepare_one_off(atoms))
            record["rewrites_b"] = list(getattr(field_b, "applied", ()))
        record["rewrites_a"] = list(getattr(field, "applied", ()))
        calls = {}
        for name, (f, t, pos) in fields.items():
            seconds = []
            for _ in range(2):  # compile, then a warm call
                torch.cuda.synchronize()
                started = time.perf_counter()
                t(pos)
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - started)
            calls[name] = dict(seconds=seconds, tiles=t.last["tiles"], slots=t.last.get("slots"),
                               max_receivers=t.last["max_receivers"], max_edges=t.last["max_edges"])
        record["calls"] = calls
        record["peak_allocated"] = int(torch.cuda.max_memory_allocated())
        save()

        if not (args.no_profile or args.split_only):
            record["profile"] = profile_call(tiled, positions)
            save()

        kept_a, g_a = capture_slots(tiled, positions)
        lanes_a = {sweep: [Lane(tiled, g_a, s, slot) for s, slot in kept_a if s == sweep] for sweep in (1, 2, 3)}
        lanes_b = {}
        if args.tf32x3:
            _, tiled_b, pos_b = fields["B"]
            kept_b, g_b = capture_slots(tiled_b, pos_b)
            lanes_b = {sweep: [Lane(tiled_b, g_b, s, slot) for s, slot in kept_b if s == sweep]
                       for sweep in (1, 2, 3)}

        streams = {} if args.split_only else {"streams": (torch.cuda.Stream(), torch.cuda.Stream())}
        if args.split_only or not args.no_split:
            try:
                made, info = green_streams()
                record["split"] = info
                if made is not None:
                    streams["split"] = tuple(made[0])
                    record["_contexts_alive"] = len(made[1])
                    keep_contexts = made[1]  # noqa: F841  (the streams need their contexts alive)
            except Exception as error:  # noqa: BLE001  (green contexts unsupported: recorded, mode skipped)
                record["split"] = dict(error=f"{type(error).__name__}: {error}"[:400])
        modes = ["serial", *streams]

        pairs = {"2+2": (lanes_a[2][0], lanes_a[2][1]), "2+1": (lanes_a[2][0], lanes_a[1][0]),
                 "2+3": (lanes_a[2][0], lanes_a[3][0]), "1+3": (lanes_a[1][0], lanes_a[3][0])}
        if lanes_b:
            pairs.update({"2+2tc": (lanes_a[2][0], lanes_b[2][1]), "2+1tc": (lanes_a[2][0], lanes_b[1][0]),
                          "2+3tc": (lanes_a[2][0], lanes_b[3][0])})
        record["lanes"] = {name: [a.label, b.label] for name, (a, b) in pairs.items()}
        alone = {}
        for lanes, tag in ((lanes_a, "A"), (lanes_b, "B")):
            for sweep, items in lanes.items():
                for i, lane in enumerate(items):
                    lane.launch()  # warm (each slot copy's shapes are compiled already)
                    alone[f"{tag}{sweep}.{i}"] = dict(label=lane.label, seconds=run_alone(lane, args.reps))
        record["alone"] = alone
        save()

        results = {name: {mode: [] for mode in modes} for name in pairs}
        for name, (a, b) in pairs.items():  # warm-up of every mode (streams, green contexts)
            for mode in modes:
                run_pair(a, b, mode, 1, streams)
        for block in range(args.blocks):
            order = modes[block % len(modes):] + modes[:block % len(modes)]
            if block % 2:
                order = order[::-1]
            for name, (a, b) in pairs.items():
                for mode in order:
                    results[name][mode].append(run_pair(a, b, mode, args.reps, streams))
            record["pairs"] = {
                name: {mode: dict(median_s=statistics.median(v), all_s=v,
                                  speedup=statistics.median(r["serial"]) / statistics.median(v))
                       for mode, v in r.items()} for name, r in results.items()}
            record["blocks_done"] = block + 1
            save()
        record.pop("_contexts_alive", None)
        save()
    print(json.dumps({name: {mode: round(v["speedup"], 3) for mode, v in r.items()}
                      for name, r in record["pairs"].items()}))
    print(json.dumps(record.get("profile", {}).get("classes", {}), default=str)[:1500])


if __name__ == "__main__":
    main()
