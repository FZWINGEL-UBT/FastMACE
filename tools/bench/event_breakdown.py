"""Where one host species-swap event's time goes (plan task 3-T20, after item 1's smoke: ~38 ms on NaCl).

One ``Simulation`` of the events side of ``tools/bench/event_speed.py`` (same model, plan, workload, 1500 K,
1 fs), warmed up with events, then two phases:

1. **Uninstrumented** (``--repeats`` rounds): ``run(every)`` twice, one event, ``run(every)`` again, each timed
   between device synchronisations. The event's wall time, and the post-event penalty: the chunk after the event
   minus the chunk before it (a restarted pipeline, a new edge size, a re-capture would show here).
2. **Instrumented** (``--repeats`` events): the pieces of the event path wrapped with a timer that synchronises
   the device before and after, so each piece gets its own device time. Times are inclusive (``canonicalise``
   contains ``flush``, ``rebuild``, ``compact``, ``_force_stage`` and ``integrator.start``); ``rest`` lines give
   what a parent spends outside its timed children. The synchronisations add a little to every piece.

The force evaluation at the canonical point is ``GraphEngine._force_stage``, the force field called directly:
it does not replay a captured graph. The record says which graphs exist and whether any capture happened.
Writes ``<output>.json``. Use an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402
import event_speed  # noqa: E402

# (parent, children) for the "rest" lines: a parent's time outside its timed children.
TREE = {
    "event": ("committed", "update"),
    "update": ("Transaction.quiesce", "Transaction.snapshot", "Transaction.apply", "Transaction.refresh",
               "Transaction.accept", "Transaction.resume"),
    # Transaction.resume's own derived.refresh (forces, stress) is in its rest: derived.refresh is also
    # Transaction.refresh's one call, and the timers do not tell the two callers apart.
    "Transaction.resume": ("canonicalise",),
    "canonicalise": ("flush", "rebuild", "compact", "_force_stage", "integrator.start", "invalidate_graphs"),
}


class Timers:
    """Inclusive times (ms) of wrapped callables, each between two device synchronisations."""

    def __init__(self, sync=lambda: None, clock=time.perf_counter):
        self.sync, self.clock = sync, clock
        self.samples = {}
        self.enabled = False

    def wrap(self, name, function):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            if not self.enabled:
                return function(*args, **kwargs)
            self.sync()
            started = self.clock()
            try:
                return function(*args, **kwargs)
            finally:
                self.sync()
                self.samples.setdefault(name, []).append((self.clock() - started) * 1e3)
        return timed

    def patch(self, owner, attribute, name=None):
        """Replace ``owner.attribute`` by its timed wrapper; returns an undo callable."""
        original = getattr(owner, attribute)
        setattr(owner, attribute, self.wrap(name or attribute, original))
        return lambda: setattr(owner, attribute, original)

    def table(self, events):
        """{name: mean ms per event} (calls per event may be > 1), with ``rest`` lines from ``TREE``."""
        per_event = {name: sum(values) / events for name, values in self.samples.items()}
        calls = {name: len(values) / events for name, values in self.samples.items()}
        for parent, children in TREE.items():
            if parent in per_event:
                per_event[f"{parent} rest"] = per_event[parent] - sum(per_event.get(child, 0.0) for child in children)
        return per_event, calls


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"])
    p.add_argument("--superset", nargs="*", default=None, metavar="SYMBOL")
    p.add_argument("--temperature-K", type=float, default=1500.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--friction-per-fs", type=float, default=0.02)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261022)
    return p.parse_args(argv)


def _median(values):
    return statistics.median(values) if values else None


def main(argv=None):
    args = parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    from fastmace.model import compile as compiling

    allocator = compiling.configure_allocator()
    import torch

    import fastmace as fm
    from fastmace import events
    from fastmace.config import EngineConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    integrator = fm.integrators.Langevin(timestep_fs=args.timestep_fs, temperature_K=args.temperature_K,
                                         friction_per_fs=args.friction_per_fs)
    meta = dict(tool="tools/bench/event_breakdown.py", task="3-T20 item 1 breakdown",
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=event_speed.source_commit(), model=args.model, checkpoint_sha256=sha,
                precision="float32", backend="hybrid", cuda_allocator=allocator,
                integrator=dataclasses.asdict(integrator), superset=args.superset, every=args.every,
                warmup=args.warmup, repeats=args.repeats, seed=args.seed, structures={})
    sync = torch.cuda.synchronize

    def timed(function):
        sync()
        started = time.perf_counter()
        function()
        sync()
        return (time.perf_counter() - started) * 1e3

    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        kinds = tuple(int(z) for z in np.unique(base.get_atomic_numbers()))
        field = ForceField(loaded, base, precision="float32", species=args.superset or None)
        sim = fm.Simulation(base, field, integrator=integrator, seed=args.seed, engine=EngineConfig(backend="hybrid"),
                            workload=Workload(max_temperature_K=args.temperature_K, species_changes=True))
        side = event_speed.SimulationSide(sim, every=args.every, kinds=kinds, seed=args.seed)
        engine = sim._engine
        for _ in range(max(1, args.warmup // args.every)):
            side.step()
        captures_before = len(engine.info()["recapture_log"])

        def event():
            committed = sim.committed()
            i, j = event_speed.swap_pair(committed.species, kinds, side.rng)
            species, momenta = event_speed.exchange(committed.species, committed.momenta, i, j)
            sim.update(species=species, momenta=momenta)

        rounds = []
        for _ in range(args.repeats):
            first = timed(lambda: sim.run(args.every))
            before = timed(lambda: sim.run(args.every))
            event_ms = timed(event)
            after = timed(lambda: sim.run(args.every))
            rounds.append(dict(first_chunk_ms=first, chunk_before_ms=before, event_ms=event_ms, chunk_after_ms=after))
        captures_uninstrumented = len(engine.info()["recapture_log"]) - captures_before

        timers = Timers(sync=sync)
        undo = [timers.patch(sim, "committed"), timers.patch(sim, "update"),
                timers.patch(engine, "canonicalise"), timers.patch(engine, "flush"),
                timers.patch(engine, "_force_stage"), timers.patch(engine, "invalidate_graphs"),
                timers.patch(engine.force_field, "rebuild"), timers.patch(engine.force_field, "compact"),
                timers.patch(engine.integrator, "start", "integrator.start"),
                timers.patch(engine.derived, "refresh", "derived.refresh"),
                timers.patch(engine.registry, "snapshot", "registry.snapshot")]
        for method in ("quiesce", "snapshot", "apply", "refresh", "accept", "resume"):
            undo.append(timers.patch(events.Transaction, method, f"Transaction.{method}"))
        timed_event = timers.wrap("event", event)
        for _ in range(args.repeats):
            sim.run(args.every)
            timers.enabled = True
            timed_event()
            timers.enabled = False
        for restore in reversed(undo):
            restore()
        per_event, calls = timers.table(args.repeats)
        steady_ms = _median([r["chunk_before_ms"] for r in rounds]) / args.every
        entry = dict(
            natoms=len(base), kinds=list(kinds), skin_A=float(engine.force_field.neighbours.skin),
            steady_ms_per_step=steady_ms,
            event_ms_median=_median([r["event_ms"] for r in rounds]),
            event_in_steps=_median([r["event_ms"] for r in rounds]) / steady_ms,
            post_event_penalty_ms_median=_median([r["chunk_after_ms"] - r["chunk_before_ms"] for r in rounds]),
            rounds=rounds, instrumented_ms_per_event=per_event, calls_per_event=calls,
            samples_ms=timers.samples, captures_while_measured=dict(
                uninstrumented=captures_uninstrumented,
                total=len(engine.info()["recapture_log"]) - captures_before),
            graphs=engine.info()["graphs"], canonical_points=engine.info()["canonical_points"],
            edge_sizes_used=engine.info()["edge_sizes_used"])
        meta["structures"][name] = entry
        print(name, f"step {steady_ms:.3f} ms, event {entry['event_ms_median']:.2f} ms "
              f"({entry['event_in_steps']:.1f} steps), "
              f"post-event penalty {entry['post_event_penalty_ms_median']:.2f} ms",
              flush=True)
        for key, value in sorted(per_event.items(), key=lambda item: -item[1]):
            print(f"  {key:28s} {value:8.3f} ms  ({calls.get(key, 0):.1f} calls)", flush=True)
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        sim.close()
        sim = side = engine = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
