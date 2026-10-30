"""Interaction-boundary checkpointing falsifier (plans/capacity.md CAP-T08, the conductor's item B).

The force call keeps every interaction's node-feature intermediates for the backward (174.5 kB per atom saved,
docs/validation/capacity-T08-inventory.md). The candidate keeps only the node state between interactions: each
interaction and its product block run under ``torch.utils.checkpoint`` (non-reentrant, no RNG stash), so their
intermediates are rebuilt in the backward. In the eager tier (``compile_mode=none``) this tool builds the default
plan's ``fm.Simulation`` for one supercell, then times force calls of the engine's force field (forward and the force
backward) and records their peak above the memory held before the call, for

- ``none``: the model as built;
- ``layer1``: interaction 1 and product 1 checkpointed;
- ``both``: interactions 0 and 1 and their products checkpointed,

in alternating rounds in one process (medians). The captured graphs were taken before any wrapper, so the force
call, not the step, is the unit. Stop rules (conductor): the peak falls by less than 40 %, or the call grows by more
than 50 %. Writes ``<output>/checkpoint-<name>-<atoms>.json``. Run inside a local GPU reservation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activation_profile  # noqa: E402
import capacity_scan  # noqa: E402

VARIANTS = {"none": (), "layer1": (1,), "both": (0, 1)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=8)
    return p.parse_args(argv)


def checkpointed(forward):
    """``forward`` (a module's bound forward, which a rewrite may have replaced) under non-reentrant checkpointing."""
    from torch.utils.checkpoint import checkpoint

    def run(*args, **kwargs):
        return checkpoint(forward, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)

    return run


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.config import EngineConfig

    name, cell = capacity_scan.base_cell(args.structure)
    atoms = cell.repeat(tuple(int(n) for n in args.repeat.lower().split("x")))
    atoms.set_pbc(True)
    spec = importlib.util.spec_from_file_location("_ab_frontends", activation_profile.TOOLS / "ab_frontends.py")
    frontends = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontends)
    frontends.thermalize(atoms, 300.0, 1)
    sim = fm.Simulation(atoms, fm.ForceField(fm.load(args.model), atoms, precision="float32"), seed=1,
                        integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                           friction_per_fs=0.002),
                        engine=EngineConfig(compile_mode="none"))
    sim.run(steps=2)
    engine = sim._engine
    model = engine.force_field.model
    model = getattr(model, "_orig_mod", model)
    layers = [(model.interactions[i], model.products[i]) for i in range(len(model.interactions))]
    positions = engine.state.positions
    reference = None

    # Each module's forward as built: the rewrites install instance-level forwards on some of them.
    built = {id(module): (module, module.__dict__.get("forward"), module.forward) for pair in layers for module in pair}

    def install(indices):
        for i, pair in enumerate(layers):
            for module in pair:
                _, own, bound = built[id(module)]
                if i in indices:
                    module.forward = checkpointed(bound)
                elif own is not None:
                    module.forward = own
                else:
                    module.__dict__.pop("forward", None)

    def call():
        out = engine.force_field(positions)
        forces = out["forces"] if isinstance(out, dict) else out[1] if isinstance(out, tuple) else out
        return forces

    results = {key: dict(ms=[], peak=[]) for key in VARIANTS}
    errors = {}
    torch.cuda.synchronize()
    for round_ in range(args.rounds):
        keys = list(VARIANTS) if round_ % 2 == 0 else list(reversed(VARIANTS))
        for key in keys:
            install(VARIANTS[key])
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            forces = call()
            stop.record()
            stop.synchronize()
            results[key]["ms"].append(start.elapsed_time(stop))
            results[key]["peak"].append(torch.cuda.max_memory_allocated() - before)
            if round_ == 0:
                value = forces.detach().clone() if torch.is_tensor(forces) else None
                if key == "none":
                    reference = value
                elif reference is not None and value is not None:
                    errors[key] = float((value - reference).abs().max())
            del forces
    install(())
    summary = {key: dict(ms=statistics.median(v["ms"]), peak_bytes=statistics.median(v["peak"]))
               for key, v in results.items()}
    base = summary["none"]
    for key, value in summary.items():
        value["peak_ratio"] = value["peak_bytes"] / base["peak_bytes"]
        value["time_ratio"] = value["ms"] / base["ms"]
    record = dict(tool="checkpoint_falsifier", structure=name, natoms=len(atoms),
                  edges=int(engine.force_field.edge_count), gpu=torch.cuda.get_device_name(), rounds=args.rounds,
                  summary=summary, force_max_abs_difference=errors, raw=results)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"checkpoint-{name}-{len(atoms)}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{name} {len(atoms)} atoms, {record['edges']} edges, eager force calls ({path})")
    for key, value in summary.items():
        print(f"  {key:7s} {value['ms']:8.2f} ms ({value['time_ratio']:.3f}x)  peak "
              f"{value['peak_bytes'] / 2**30:6.3f} GiB ({value['peak_ratio']:.3f}x)  "
              f"max |dF| {errors.get(key, 0.0):.2e}")
    sim.close()
    sys.stdout.flush()  # os._exit skips the buffers of a redirected stdout
    os._exit(0)


if __name__ == "__main__":
    main()
