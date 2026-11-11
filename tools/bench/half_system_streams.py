"""Idea 2 of 11 Nov 2026 ("two half-systems out of phase"), first falsifier: two half-size force graphs on two
streams against back to back, and against the full system's graph.

The proposal splits the atoms into halves A and B so that while B's edge kernel fills the GPU, A's atom chain (GEMMs,
contraction, glue) runs on a second graph branch. Its upper bound is two *independent* half-size force graphs
replayed concurrently on two streams: the real split couples the halves at every layer (interaction 2 of A reads the
layer-1 features of B's senders), so it can only overlap less. If the concurrent pair is not clearly faster than the
same pair back to back, the idea has no room.

Engines: the minimal planner's default plan of the GPU (as tools/step_breakdown.py builds it) for the full structure
and for two copies of the half structure (the same cell halved along c; different velocity seeds). Each engine runs
--equilibrate-steps MD steps so its force graph ("B") is captured, then the graphs are replayed in alternating rounds:

- full:       the full system's graph;
- serial:     A then B on one stream;
- concurrent: A on stream 1, B on stream 2 (both wait on the round's start, the round waits on both): in phase;
- offset (review retry, 11 Nov 2026): a block of --replays replays of A on stream 1 and of B on stream 2, B delayed
  once at the block's start by half a half-system graph (a calibrated ``torch.cuda._sleep``), then both free-running:
  equal graphs stay half a step apart, so B's edge kernels meet A's atom chain. ``offset`` is per replay pair
  including the one delay; ``offset_net`` subtracts the delay.

    python tools/bench/half_system_streams.py --checkpoint mace-omat-0-medium.model --output DIR \\
        [--full SrTiO3@10x10x10 --half SrTiO3@10x10x5]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ab_step_time  # noqa: E402
import step_breakdown  # noqa: E402


def engine_for(args, spec, seed):
    import numpy as np
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    _, atoms = ab_step_time.read_structure(spec, None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(seed))
    Stationary(atoms)
    ns = SimpleNamespace(model=args.model, checkpoint=args.checkpoint, backend="hybrid", rewrites=None, skin=None,
                         temperature_K=300.0, timestep_fs=None)
    engine, plan = step_breakdown.build_engine(ns, atoms)
    for _ in range(args.equilibrate_steps):
        engine.step()
    engine.flush()
    edges = int(engine.force_field.data["edge_index"].shape[1])
    key = step_breakdown.force_graph_key(engine, edges)
    if key is None:
        raise SystemExit(f"no captured force graph for {spec}")
    return engine, engine._graphs[key], dict(spec=spec, atoms=len(atoms), padded_edges=edges, graph=str(key),
                                             options=plan["options"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--full", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--half", default="results/loup-a100-20261103/SrTiO3.cif@10x10x5")
    p.add_argument("--equilibrate-steps", type=int, default=50)
    p.add_argument("--rounds", type=int, default=21)
    p.add_argument("--replays", type=int, default=20, help="replays per timed block")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    full, g_full, info_full = engine_for(args, args.full, 1)
    a, g_a, info_a = engine_for(args, args.half, 2)
    b, g_b, info_b = engine_for(args, args.half, 3)
    main_stream = torch.cuda.current_stream()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    def run_full():
        g_full.replay()

    def run_serial():
        g_a.replay()
        g_b.replay()

    def run_concurrent():
        start = torch.cuda.Event()
        start.record(main_stream)
        for s, g in ((s1, g_a), (s2, g_b)):
            s.wait_event(start)
            with torch.cuda.stream(s):
                g.replay()
        for s in (s1, s2):
            main_stream.wait_stream(s)

    def sleep_cycles_per_us():
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(main_stream)
        torch.cuda._sleep(10_000_000)
        e1.record(main_stream)
        torch.cuda.synchronize()
        return 10_000_000 / (e0.elapsed_time(e1) * 1000.0)

    rate = sleep_cycles_per_us()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        g_a.replay()
    torch.cuda.synchronize()
    delay_us = (time.perf_counter() - t0) / 20 * 1e6 / 2  # half of one half-system graph
    delay_cycles = int(delay_us * rate)

    def run_offset_block():
        start = torch.cuda.Event()
        start.record(main_stream)
        s1.wait_event(start)
        s2.wait_event(start)
        with torch.cuda.stream(s2):
            torch.cuda._sleep(delay_cycles)
            for _ in range(args.replays):
                g_b.replay()
        with torch.cuda.stream(s1):
            for _ in range(args.replays):
                g_a.replay()
        for s in (s1, s2):
            main_stream.wait_stream(s)

    sides = dict(full=run_full, serial=run_serial, concurrent=run_concurrent, offset=run_offset_block)

    def timed(fn):
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(main_stream)
        if fn is run_offset_block:
            fn()
        else:
            for _ in range(args.replays):
                fn()
        e1.record(main_stream)
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000.0 / args.replays

    for fn in sides.values():  # warm-up
        timed(fn)
    times = {k: [] for k in sides}
    names = list(sides)
    for r in range(args.rounds):
        for k in (names if r % 2 == 0 else names[::-1]):
            times[k].append(timed(sides[k]))
    med = {k: statistics.median(v) for k, v in times.items()}
    med["offset_net"] = med["offset"] - delay_us / args.replays
    record = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, commit=ab_step_time.fastmace_commit(),
                  created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rounds=args.rounds,
                  replays=args.replays, full=info_full, half_a=info_a, half_b=info_b, median_us=med, times_us=times,
                  concurrent_over_serial=med["concurrent"] / med["serial"], serial_over_full=med["serial"] / med["full"],
                  concurrent_over_full=med["concurrent"] / med["full"], offset_delay_us=delay_us,
                  sleep_cycles_per_us=rate, offset_over_serial=med["offset"] / med["serial"],
                  offset_net_over_serial=med["offset_net"] / med["serial"],
                  offset_net_over_concurrent=med["offset_net"] / med["concurrent"])
    (args.output / "half_system_streams.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("times_us", "full", "half_a", "half_b")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
