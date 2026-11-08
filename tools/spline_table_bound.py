"""Bound: how much of the SPLINE_W forward and the w' kernel is the spline table read (8 November 2026).

The external "breakthrough" report of 8 November ranks a 64-channel h(r) spline (proposal 1) and undirected edge pairs
in w' (proposal 3) on the claim that both kernels are bound by streaming the spline table (20 KB per edge). This tool
captures both kernels' real arguments from one eager force call after equilibration (the default plan of the GPU,
or the one ``--constant`` forces), then times each kernel in alternating rounds with

- ``real``: the captured arguments;
- ``one-slab``: every edge inside the cutoff reads the same table slab (pair 0, one interval; padded edges unchanged),
  so the table read costs what an L1-resident slab costs. The outputs are wrong; only the time counts.

The difference is the most any change of the table traffic can save in that kernel (proposal 1's whole table saving,
proposal 3's table share is at most half of it). Timings with an L2 flush before each launch and without.

    python tools/spline_table_bound.py --checkpoint mace-omat-0-medium.model --output DIR \
        [--constant fastmace.kernels.oeq_streaming.spline_forward:SPLINE_W=True ...]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402


def capture(args):
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
    from fastmace.kernels.oeq_streaming import spline_forward, wprime_backward
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    classes = {"forward": spline_forward.SplineForward, "wprime": wprime_backward.WPrimeBackward}
    edge_arg = {"forward": 1, "wprime": 1}  # sh [E, 16]
    captured, recording = {}, [False]
    originals = {name: cls.__call__ for name, cls in classes.items()}

    def recorder(name, original):
        def call(self, *a):
            edges = a[edge_arg[name]].shape[0]
            largest = captured[name][1][edge_arg[name]].shape[0] if name in captured else -1
            if recording[0] and not torch.cuda.is_current_stream_capturing() and edges > largest:
                captured[name] = (self, [t.clone() if isinstance(t, torch.Tensor)
                                         else [u.clone() for u in t] if isinstance(t, list) else t for t in a])
            return original(self, *a)
        return call

    for name, cls in classes.items():
        cls.__call__ = recorder(name, originals[name])
    try:
        planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"), neighbours=NeighbourConfig(skin=0.5))
        sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
        _, base = ab_step_time.read_structure(args.structure, None)
        field = ForceField(loaded, base, precision="float32", rewrites="auto")
        integrator = IntegratorConfig(temperature_K=300.0, timestep_fs=1.0)
        plan = planner.plan(field, base, integrator, Workload(max_temperature_K=300.0), planner.hardware())
        engine = planner.build(plan, field, base.copy(), temperature_K=300.0)
        engine.prepare()
        for _ in range(args.equilibrate_steps):  # thermal disorder: the ideal lattice touches few slabs
            engine.step()
        engine.flush()
        torch.cuda.synchronize()
        recording[0] = True
        engine.invalidate_graphs("spline_table_bound: eager call for the capture")
        engine.step()
        engine.flush()
        torch.cuda.synchronize()
    finally:
        for name, cls in classes.items():
            cls.__call__ = originals[name]
    missing = [n for n in classes if n not in captured]
    if missing:
        raise SystemExit(f"no eager call of {missing} (is SPLINE_W on?)")
    del engine, field, plan, loaded
    gc.collect()
    torch.cuda.empty_cache()
    return captured, sha


def one_slab(name, args, r_max):
    """The arguments with every edge inside the cutoff on one table slab (pair 0, the first such edge's interval)."""
    import torch
    args = list(args)
    i_r, i_pair = (4, 5) if name == "forward" else (8, 9)
    r, pair = args[i_r], args[i_pair]
    if r.dtype == torch.int32 and r.dim() == 2:  # SHARED_LOCATOR records int4(slab, t, 1/h, inside)
        loc = r.clone()
        inside = loc[:, 3] != 0
        first = int(torch.nonzero(inside)[0])
        loc[inside] = loc[first]
        args[i_r] = loc
        return args, int(inside.sum())
    inside = r < r_max
    first = int(torch.nonzero(inside)[0])
    rr = r.clone()
    rr[inside] = r[first]
    pp = pair.clone()
    pp[inside] = pair[first]
    args[i_r], args[i_pair] = rr, pp
    return args, int(inside.sum())


def slab_sorted(args):
    """The forward's arguments with every receiver's edges reordered by (pair, length): the same message up to the
    summation order, so co-resident receivers walk the same slabs at about the same time."""
    import torch
    args = list(args)
    sh, senders, row_ptr, r, pair = args[1], args[2], args[3], args[4], args[5]
    counts = (row_ptr[1:] - row_ptr[:-1]).long()
    receiver = torch.repeat_interleave(torch.arange(counts.numel(), device=r.device), counts)
    if r.dtype == torch.int32 and r.dim() == 2:  # SHARED_LOCATOR records: slab, then t
        within = r[:, 0].double() * 2.0 + r[:, 1].view(torch.float32).double()
    else:
        within = pair.double() * 16.0 + r.reshape(-1).double()
    edges = receiver.numel()
    key = receiver.double() * 1e3 + within[:edges] / 1e2
    order = torch.argsort(key, stable=True)
    if r.shape[0] > edges:  # rows past row_ptr's end stay where they are
        order = torch.cat([order, torch.arange(edges, r.shape[0], device=r.device)])
    args[1], args[2], args[4], args[5] = sh[order], senders[order], r[order], pair[order]
    return args


def species_blocks(kernel, args, r_max):
    """BLOCK_ORDER's variant of a captured forward: the same kernel rendered with an order argument, and the receivers
    grouped by species (the receiver's species from its first edge's pair, sender x S + receiver; H200 tables are
    not SYMMETRIC_PAIRS). Returns (kernel, order, seconds of the order's construction per call) or None."""
    import torch

    from fastmace.kernels.oeq_streaming.spline_forward import SplineForward, receiver_order
    row_ptr, r, pair, table = args[3], args[4], args[5], args[6]
    species = int(round((table.shape[0] / kernel.intervals) ** 0.5))
    counts = (row_ptr[1:] - row_ptr[:-1]).long()
    edges = int(counts.sum())
    receiver = torch.repeat_interleave(torch.arange(counts.numel(), device=r.device), counts)
    inside = r.reshape(-1)[:edges] < r_max  # padding and edges past the cutoff carry no species information
    own = pair[:edges].long() % species
    receiver_species = torch.full((counts.numel(),), species, dtype=torch.long, device=r.device)
    receiver_species.scatter_reduce_(0, receiver[inside], own[inside], "amin", include_self=True)
    if not torch.equal(receiver_species[receiver[inside]], own[inside]):
        print("species-blocks: pair % S differs within a receiver (symmetric tables?); skipped", flush=True)
        return None
    first = row_ptr[:-1].long().clamp(0, pair.numel() - 1)
    ordered = SplineForward(*kernel.init_args, block_order=True)
    order = receiver_order(receiver_species)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        receiver_order(torch.where(counts > 0, pair[first].long() % species, species))  # the per-call cost
    end.record()
    end.synchronize()
    return ordered, order, start.elapsed_time(end) / 20 * 1000.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--constant", nargs="*", default=[], metavar="MODULE:NAME=VALUE")
    p.add_argument("--rounds", type=int, default=21)
    p.add_argument("--equilibrate-steps", type=int, default=300)
    p.add_argument("--flush-l2-mb", type=int, default=256)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from ab_rewrites import constant_overrides

    constants = constant_overrides(args.constant)
    for (module, name), value in constants.items():
        setattr(sys.modules[module], name, value)
    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=True)
    captured, sha = capture(args)
    flush = torch.empty(args.flush_l2_mb * 2 ** 20 // 4, dtype=torch.float32, device="cuda")
    result = dict(tool="spline_table_bound", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  fastmace_commit=ab_step_time.fastmace_commit(), checkpoint_sha256=sha, rounds=args.rounds,
                  constants={f"{m}:{n}": v for (m, n), v in constants.items()}, kernels={})
    r_max = float(captured["wprime"][1][16])  # the cutoff; padded edges sit at or beyond it
    for name, (kernel, real) in captured.items():
        slab, inside = one_slab(name, real, r_max)
        variants = {"real": real, "one-slab": slab}
        kernels = {}  # a variant's own kernel and keyword arguments (BLOCK_ORDER)
        if name == "forward":
            variants["slab-sorted"] = sorted_args = slab_sorted(real)
            blocks = species_blocks(kernel, real, r_max)
            if blocks is not None:
                ordered, order, build_us = blocks
                print(f"species-blocks: the order costs {build_us:.1f} us per call (eager torch)", flush=True)
                variants["species-blocks"] = real
                variants["slab-sorted+species-blocks"] = sorted_args
                kernels["species-blocks"] = kernels["slab-sorted+species-blocks"] = (ordered, dict(order=order))

        def run(label, a):
            k, extra = kernels.get(label, (kernel, {}))
            k(*a, **extra)

        def launch(a, label="real"):
            if name == "wprime":
                a[6].zero_()
            run(label, a)

        times = {label: {"flush": [], "warm": []} for label in variants}
        for _ in range(3):
            for label, a in variants.items():
                launch(a, label)
        torch.cuda.synchronize()
        if name == "forward":  # the same message up to summation order (bitwise for species-blocks alone)
            def message(label):
                a = variants[label]
                o = [t.clone().zero_() for t in a[7]]
                k, extra = kernels.get(label, (kernel, {}))
                k(*a[:7], o, **extra)
                return o
            base = message("real")
            for label in [v for v in variants if v not in ("real", "one-slab")]:
                got = message(label)
                torch.cuda.synchronize()
                worst = max(float((u - v).abs().max() / v.abs().max().clamp_min(1e-30)) for u, v in zip(got, base))
                print(f"forward {label} against real: max|d|/max|a| {worst:.2e}", flush=True)
        labels = list(variants)
        for k in range(args.rounds):
            ordered = labels if k % 2 == 0 else labels[::-1]
            for label in ordered:
                a = variants[label]
                for mode in ("flush", "warm"):
                    if mode == "flush":
                        flush.zero_()
                    else:
                        launch(a, label)  # the same launch once before: its table lines are in L2
                    if name == "wprime":
                        a[6].zero_()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    run(label, a)
                    end.record()
                    end.synchronize()
                    times[label][mode].append(start.elapsed_time(end) * 1000.0)
        med = {label: {mode: statistics.median(v) for mode, v in d.items()} for label, d in times.items()}
        entry = dict(edges=int(real[1].shape[0]), edges_inside=inside, median_us=med,
                     over_real={label: {mode: med[label][mode] / med["real"][mode] for mode in ("flush", "warm")}
                                for label in variants},
                     registers=getattr(kernel, "registers", None))
        result["kernels"][name] = entry
        print(f"{name}: edges {entry['edges']} (inside {inside}); " + "; ".join(
            f"{label} flush {med[label]['flush']:.1f} / warm {med[label]['warm']:.1f} us "
            f"(x {entry['over_real'][label]['flush']:.4f} / {entry['over_real'][label]['warm']:.4f})"
            for label in variants), flush=True)
    (args.output / "spline_table_bound.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
