"""Host-side timing of NaCl's fixed costs per MD step (plan task OPT2-T04; measurement only).

The engine is ``step_breakdown.build_engine``'s (the minimal planner's default plan, plus the OPT-T08
records the round's baseline switches on). After --equilibrate-steps at --temperature-K, --steps MD steps
are timed, with the host time of each engine part summed by wrapping the instance's methods:
``_rebuild`` (with its host sync), ``_validate_pipelined`` (the wait on the previous step's flags),
``integrator.prepare_step`` (the noise upload) and the graph replays. Prints one JSON line.

    python tools/opt2_t04_probe.py --checkpoint MODEL [--temperature-K 1000] [--steps 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import fastmace  # noqa: F401  (before torch)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402
import step_breakdown  # noqa: E402

BASELINE = ["species_tables=on", "fold_readout_tail=on", "fold_product_linear=on"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--equilibrate-steps", type=int, default=300)
    p.add_argument("--rewrites", nargs="*", default=BASELINE)
    args = p.parse_args(argv)
    breakdown = step_breakdown.parse_args(["--output", "/nonexistent", "--checkpoint", str(args.checkpoint),
                                           "--structure", args.structure, "--temperature-K", str(args.temperature_K),
                                           "--rewrites", *args.rewrites])
    ab_step_time.prepare_fastmace_hybrid()
    import numpy as np
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    _, atoms = ab_step_time.read_structure(args.structure, None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(20261007))
    Stationary(atoms)
    engine, plan = step_breakdown.build_engine(breakdown, atoms)
    for _ in range(args.equilibrate_steps):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()

    host = defaultdict(float)
    calls = defaultdict(int)

    def timed(owner, name, label):
        original = getattr(owner, name)

        def wrapper(*a, **k):
            started = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                host[label] += time.perf_counter() - started
                calls[label] += 1

        setattr(owner, name, wrapper)

    timed(engine, "_rebuild", "rebuild")
    timed(engine, "_validate_pipelined", "validate")
    timed(engine.integrator, "prepare_step", "prepare_step")
    timed(engine, "_run", "replays")
    before = dict(rebuilds=engine.rebuilds, redos=engine.pipelined_redos, early=engine.pipelined_early_rebuilds,
                  rebuild_redos=engine.pipelined_rebuild_redos)
    started = time.perf_counter()
    for _ in range(args.steps):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    counters = dict(rebuilds=engine.rebuilds - before["rebuilds"], redos=engine.pipelined_redos - before["redos"],
                    early_rebuilds=engine.pipelined_early_rebuilds - before["early"],
                    rebuild_redos=engine.pipelined_rebuild_redos - before["rebuild_redos"])
    per_step_ms = {label: 1e3 * seconds / args.steps for label, seconds in host.items()}
    per_call_ms = {label: 1e3 * host[label] / calls[label] for label in host if calls[label]}
    print(json.dumps(dict(structure=args.structure, temperature_K=args.temperature_K, steps=args.steps,
                          wall_ms_per_step=1e3 * wall / args.steps, host_ms_per_step=per_step_ms,
                          host_ms_per_call=per_call_ms, calls=dict(calls), counters=counters,
                          pipelined=engine.pipelined, speculative=engine.speculative,
                          graphs=[str(k) for k in engine._graphs], gpu=torch.cuda.get_device_name())), flush=True)


if __name__ == "__main__":
    main()
