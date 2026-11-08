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

        def launch(a):
            if name == "wprime":
                a[6].zero_()
            kernel(*a)

        times = {"real": {"flush": [], "warm": []}, "one-slab": {"flush": [], "warm": []}}
        for _ in range(3):
            launch(real)
            launch(slab)
        torch.cuda.synchronize()
        for k in range(args.rounds):
            order = (("real", real), ("one-slab", slab)) if k % 2 == 0 else (("one-slab", slab), ("real", real))
            for label, a in order:
                for mode in ("flush", "warm"):
                    if mode == "flush":
                        flush.zero_()
                    else:
                        launch(a)  # the same launch once before: its table lines are in L2
                    if name == "wprime":
                        a[6].zero_()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    kernel(*a)
                    end.record()
                    end.synchronize()
                    times[label][mode].append(start.elapsed_time(end) * 1000.0)
        med = {label: {mode: statistics.median(v) for mode, v in d.items()} for label, d in times.items()}
        entry = dict(edges=int(real[1].shape[0]), edges_inside=inside, median_us=med,
                     one_slab_over_real={mode: med["one-slab"][mode] / med["real"][mode] for mode in ("flush", "warm")},
                     registers=getattr(kernel, "registers", None))
        result["kernels"][name] = entry
        print(f"{name}: edges {entry['edges']} (inside {inside}); real flush {med['real']['flush']:.1f} / warm "
              f"{med['real']['warm']:.1f} us; one-slab flush {med['one-slab']['flush']:.1f} / warm "
              f"{med['one-slab']['warm']:.1f} us; ratio {entry['one_slab_over_real']}", flush=True)
    (args.output / "spline_table_bound.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
