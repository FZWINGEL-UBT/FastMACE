"""Phase 3 acceptance item 1 (plan task 3-T20): species-change events against fixed species, one process.

Two ``Simulation`` sides of one structure and model, MH-1 hybrid float32 Langevin NVT by default, at a
1500 K target and 1 fs:

  fixed   ``Workload(max_temperature_K=T)``: no species change
  events  ``Workload(max_temperature_K=T, species_changes=True)``; after every ``--every`` steps one host
          event, ``Simulation.update(species=..., momenta=...)``, swaps the species of one atom of each of the
          structure's two species with ``exchange`` momenta (SPEC §10.4: the two atoms swap velocities with
          their species) and no energy evaluation. The event is the host path: quiesce, apply, refresh the
          derived buffers, then a canonical point (neighbour list rebuilt, forces evaluated once,
          ``integrator.start``); its cost counts. The pairs come from a seeded stream, the same in every run.

Each side advances with ``Simulation.run(every)``, so a timed unit is one chunk of ``every`` steps (plus its
event on the events side); ``--warmup`` and ``--block`` must be whole chunks, so every block holds the same
number of events. Both sides are built and warmed up first, then 12 blocks of each alternate
(``fastmace.bench.ab.run_ab``; build order ``--order``, pattern ``--pattern``). The result is the rate
ratio median(fixed) / median(events), and the run passes at >= 0.90 (SPEC §16 row 3). It is invalid when
anything compiled, any graph was captured (``engine.info()["recapture_log"]``) or another process used the
GPU while timed; ``--settle N`` keeps warming up (untimed, at most ``--settle-max`` steps) until N chunks in a
row add no capture, since at 1500 K new edge sizes keep appearing. ``--superset`` gives both sides that species
superset (``fm.ForceField(species=)``). Each side's skin and headroom (rule 11, 3-T05: it sizes the events
side, which declares species changes) and edge sizes are recorded.

Writes ``<output>.json``. Use an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ab_step_time  # noqa: E402

SIDES = ("fixed", "events")
PASS_RATIO = 0.90


def swap_pair(species, kinds, rng):
    """(i, j): one atom of species ``kinds[0]`` and one of ``kinds[1]``, uniform, from ``rng``."""
    first, second = (np.flatnonzero(np.asarray(species) == kind) for kind in kinds)
    if not len(first) or not len(second):
        raise ValueError(f"no atom of species {kinds[0] if not len(first) else kinds[1]} to swap")
    return int(first[rng.integers(len(first))]), int(second[rng.integers(len(second))])


def exchange(species, momenta, i, j):
    """New (species, momenta) with atoms ``i`` and ``j`` swapped: ``exchange`` momenta, p_i' = p_j (§10.4)."""
    species, momenta = np.array(species, copy=True), np.array(momenta, dtype=np.float64, copy=True)
    species[[i, j]] = species[[j, i]]
    momenta[[i, j]] = momenta[[j, i]]
    return species, momenta


class SimulationSide:
    """A ``Simulation`` as a ``fastmace.bench.step_time.TimedEngine`` whose step is ``every`` MD steps.

    With ``kinds`` (two atomic numbers), each step ends with one species-swap event through
    ``Simulation.update``."""

    def __init__(self, sim, *, every, kinds, seed, torch=None):
        self.sim, self.every, self.kinds, self.torch = sim, every, kinds, torch
        self.rng = np.random.default_rng(seed)
        self.events = 0

    def step(self):
        self.sim.run(self.every)
        if self.kinds is None:
            return
        committed = self.sim.committed()
        i, j = swap_pair(committed.species, self.kinds, self.rng)
        species, momenta = exchange(committed.species, committed.momenta, i, j)
        self.sim.update(species=species, momenta=momenta)
        self.events += 1

    def prepare_block(self):
        pass

    def finish_block(self):
        pass  # run() and update() end on a committed step

    def synchronize(self):
        if self.torch is not None:
            self.torch.cuda.synchronize()


def source_commit(environ=os.environ):
    """This checkout's commit: ``ab_step_time.fastmace_commit``, else ``$FASTMACE_SOURCE_COMMIT`` (an app
    worktree on a Windows drive, whose ``.git`` file names a Windows path that WSL's git cannot follow)."""
    return ab_step_time.fastmace_commit() or environ.get("FASTMACE_SOURCE_COMMIT") or None


def settle(sides, captures, *, quiet, limit):
    """Untimed chunks on every side until ``quiet`` chunks in a row add no graph capture (``captures()`` gives
    {side: captures logged}), at most ``limit`` chunks; returns the chunks run. At 1500 K the edge count keeps
    growing past a short warm-up, and a capture while timed invalidates the run (3-T20 breakdown)."""
    if quiet <= 0:
        return 0
    last, calm, chunks = captures(), 0, 0
    while calm < quiet and chunks < limit:
        for side in sides.values():
            side.step()
        chunks += 1
        now = captures()
        calm, last = (calm + 1, last) if now == last else (0, now)
    return chunks


def neighbour_record(engine):
    """The skin and headroom the plan gave the engine (rule 11, 3-T05) and its edge sizes."""
    neighbours, config = engine.force_field.neighbours, engine.neighbour_config
    return dict(skin_A=float(neighbours.skin), capacity_headroom=neighbours.headroom,
                edge_bucket_fraction=config.edge_bucket_fraction, edge_bucket=int(engine.edge_bucket),
                edge_capacity=int(engine.force_field.edge_capacity), recapture_budget=engine.recapture_budget)


def chunk_protocol(args):
    """The timing protocol in chunks of ``args.every`` steps."""
    from fastmace.bench.step_time import TimingProtocol

    return TimingProtocol(warmup=args.warmup // args.every, block=args.block // args.every, blocks=args.blocks)


def summarise(medians, *, every, threshold=PASS_RATIO):
    """Per-step medians (ms), the rate ratio fixed/events and the verdict, from medians per chunk."""
    rate_ratio = medians["fixed"] / medians["events"]
    return dict(ms_per_step={side: medians[side] / every for side in SIDES}, rate_ratio=rate_ratio,
                threshold=threshold, **{"pass": bool(rate_ratio >= threshold)})


def validity(captures_before, captures_after, *, recompiled, others):
    """(valid, reasons): no compile, no graph capture on either side and no other GPU process while timed."""
    reasons = [f"{side}: {captures_after[side] - captures_before[side]} graph capture(s) while timed"
               for side in SIDES if captures_after[side] != captures_before[side]]
    if recompiled:
        reasons.append("compiled while timed")
    if others:
        reasons.append(f"other GPU processes: {others}")
    return not reasons, reasons


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"])
    p.add_argument("--superset", nargs="*", default=None, metavar="SYMBOL",
                   help="Both sides' species superset (default: the structure's species)")
    p.add_argument("--temperature-K", type=float, default=1500.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--friction-per-fs", type=float, default=0.02)
    p.add_argument("--every", type=int, default=100, help="Steps between events")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--pattern", choices=("abab", "abba"), default="abab")
    p.add_argument("--order", choices=("ab", "ba"), default="ab", help="Build order: ab = fixed first")
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--settle", type=int, default=0, metavar="CHUNKS",
                   help="After --warmup, keep warming up until this many chunks in a row add no graph capture")
    p.add_argument("--settle-max", type=int, default=3000, metavar="STEPS", help="At most this many settle steps")
    args = p.parse_args(argv)
    if args.every < 1 or args.block % args.every or args.warmup % args.every or args.block < args.every:
        p.error(f"--warmup and --block must be whole multiples of --every ({args.every})")
    if args.settle < 0 or args.settle_max % args.every:
        p.error(f"--settle must be >= 0 and --settle-max a whole multiple of --every ({args.every})")
    return args


def _side_record(sim):
    engine = sim._engine
    info = engine.info()
    committed = sim.committed()
    natoms = len(committed.species)
    masses = committed.masses
    kinetic = 0.5 * float((committed.momenta ** 2 / masses[:, None]).sum())
    from ase import units
    return dict(**neighbour_record(engine), recaptures=info["recaptures"],
                captures_logged=len(info["recapture_log"]), neighbour_rebuilds=info["neighbour_rebuilds"],
                canonical_points=info["canonical_points"], edge_sizes_used=info["edge_sizes_used"],
                overflows_redone=info["overflows_redone"], step=committed.step,
                temperature_K=2 * kinetic / (3 * natoms * units.kB), potential_eV_atom=committed.energy / natoms,
                species_counts={int(z): int(n) for z, n in zip(*np.unique(committed.species, return_counts=True))})


def main(argv=None):
    args = parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    from fastmace.model import compile as compiling

    allocator = compiling.configure_allocator()  # before the first CUDA allocation (the engine checks it)
    import torch
    from torch._dynamo.utils import counters

    import fastmace as fm
    from fastmace.bench import ab
    from fastmace.config import EngineConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    protocol = chunk_protocol(args)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    integrator = fm.integrators.Langevin(timestep_fs=args.timestep_fs, temperature_K=args.temperature_K,
                                         friction_per_fs=args.friction_per_fs)
    order = ["fixed", "events"] if args.order == "ab" else ["events", "fixed"]
    meta = dict(tool="tools/bench/event_speed.py", task="3-T20 item 1",
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=source_commit(), model=args.model, checkpoint_sha256=sha,
                precision="float32", backend="hybrid", cuda_allocator=allocator,
                integrator=dataclasses.asdict(integrator),
                superset=args.superset, every=args.every, seed=args.seed,
                protocol=dict(warmup_steps=args.warmup, block_steps=args.block, blocks=args.blocks,
                              settle_chunks=args.settle, settle_max_steps=args.settle_max),
                pattern=args.pattern, order=order, load=list(os.getloadavg()), structures={})
    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        kinds = tuple(int(z) for z in np.unique(base.get_atomic_numbers()))
        if len(kinds) != 2:
            raise SystemExit(f"{name}: the swap event needs exactly two species, not {kinds}")
        sides, sims, setup = {}, {}, {}
        for side in order:
            started = time.perf_counter()
            field = ForceField(loaded, base, precision="float32", species=args.superset or None)
            workload = Workload(max_temperature_K=args.temperature_K, species_changes=side == "events")
            sims[side] = fm.Simulation(base, field, integrator=integrator, workload=workload, seed=args.seed,
                                       engine=EngineConfig(backend="hybrid"))
            sides[side] = SimulationSide(sims[side], every=args.every, kinds=kinds if side == "events" else None,
                                         seed=args.seed, torch=torch)
            setup[side] = time.perf_counter() - started
        from fastmace.bench import step_time
        for side in order:  # warm up outside the compile-counter window (new edge sizes compile here)
            step_time.warm_up(sides[side], protocol.warmup)

        def logged():
            return {side: len(sims[side]._engine.info()["recapture_log"]) for side in SIDES}

        settled = settle({side: sides[side] for side in order}, logged, quiet=args.settle,
                         limit=args.settle_max // args.every)
        captures = logged()
        counted = ab_step_time.compile_counters(counters)
        events_before = sides["events"].events
        result = ab.run_ab(sides, dataclasses.replace(protocol, warmup=0), pattern=args.pattern)
        changes, recompiled = ab_step_time.compile_changes(counted, ab_step_time.compile_counters(counters))
        after = {side: len(sims[side]._engine.info()["recapture_log"]) for side in SIDES}
        others = [pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()]
        valid, reasons = validity(captures, after, recompiled=recompiled, others=others)
        medians = {side: result.statistics[side].median_ms for side in SIDES}
        summary = summarise(medians, every=args.every)
        blocks = {side: result.statistics[side].block_ms for side in SIDES}
        entry = dict(natoms=len(base), kinds=list(kinds), settle_steps=settled * args.every, ab=result.as_dict(),
                     **summary,
                     pair_rate_ratios=[f / e for f, e in zip(blocks["fixed"], blocks["events"])],
                     events_timed=sides["events"].events - events_before, events_total=sides["events"].events,
                     setup_s=setup, compile_during_timing=dict(changes=changes, recompiled=recompiled),
                     sides={side: _side_record(sims[side]) for side in SIDES},
                     plans={side: sims[side].plan.describe() for side in SIDES},
                     load_after=list(os.getloadavg()), other_gpu_processes=others, valid=valid,
                     invalid_reasons=reasons)
        meta["structures"][name] = entry
        print(name, " ".join(f"{s} {summary['ms_per_step'][s]:.4f} ms/step" for s in SIDES),
              f"rate fixed/events {summary['rate_ratio']:.4f}", "PASS" if summary["pass"] else "FAIL",
              "valid" if valid else f"INVALID {reasons}", flush=True)
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        for sim in sims.values():
            sim.close()
        sides = sims = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
