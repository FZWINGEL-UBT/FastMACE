"""What one MD force evaluation keeps alive at its memory peak (plans/capacity.md, CAP-T07 and CAP-T08).

Builds the default plan's ``fm.Simulation`` for one supercell, as ``capacity_scan.py point`` does, then records the
caching allocator's history (``torch.cuda.memory._record_memory_history``, Python stacks) over one eager force call
of the engine's force field (the stage the warm-up runs before each capture). Replaying the allocation and free
events finds the moment of the largest live total; the blocks alive then are grouped by the innermost frame of the
model (MACE, e3nn, cuEquivariance, OEq or FastMACE) that allocated them, with their shapes as bytes per edge and per
atom. Prints the top sites and writes ``<output>/profile-<name>-<precision>-<atoms>.json``.

``--engine compile_mode=none`` keeps the model eager, so the frames name the operations rather than Inductor's
generated code. Run inside a local GPU reservation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity_scan  # noqa: E402

PACKAGES = ("fastmace", "mace", "e3nn", "cuequivariance", "openequivariance", "torch/_inductor", "torch/nn/functional")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--engine", nargs="*", default=[], metavar="FIELD=VALUE", help="EngineConfig overrides")
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE", help="RewriteConfig overrides")
    p.add_argument("--top", type=int, default=40)
    return p.parse_args(argv)


def _site(frames):
    """The innermost frame inside the model's packages, and the frames above it (file:line function)."""
    def short(frame):
        name = frame["filename"]
        for package in PACKAGES:
            at = name.find("/" + package)
            if at >= 0:
                name = name[at + 1:]
                break
        return f"{name}:{frame['line']} {frame['name']}"

    inner = [f for f in frames if any("/" + p in f["filename"] for p in PACKAGES)
             and "torch/cuda" not in f["filename"]]
    if not inner:
        return "unattributed", []
    return short(inner[0]), [short(f) for f in inner[1:5]]


def live_at_peak(trace):
    """Replay the device trace: (peak bytes, {address: (size, frames)} alive at the peak)."""
    live, total, best, best_live = {}, 0, 0, {}
    for event in trace:
        action, addr, size = event["action"], event["addr"], event["size"]
        if action == "alloc":
            live[addr] = (size, event.get("frames", []))
            total += size
            if total > best:
                best, best_live = total, dict(live)
        elif action in ("free_requested", "free_completed"):
            if addr in live and action == "free_requested":
                total -= live.pop(addr)[0]
    return best, best_live


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.config import EngineConfig, RewriteConfig

    name, cell = capacity_scan.base_cell(args.structure)
    repeat = tuple(int(n) for n in args.repeat.lower().split("x"))
    atoms = cell.repeat(repeat)
    atoms.set_pbc(True)
    frontends_spec = importlib.util.spec_from_file_location("_ab_frontends", TOOLS / "ab_frontends.py")
    frontends = importlib.util.module_from_spec(frontends_spec)
    frontends_spec.loader.exec_module(frontends)
    frontends.thermalize(atoms, 300.0, 1)
    engine_options = {k: capacity_scan._value(v) for k, v in (item.split("=", 1) for item in args.engine)}
    rewrites = dict(item.split("=", 1) for item in args.rewrites)
    field = fm.ForceField(fm.load(args.model), atoms, precision=args.precision,
                          **({"rewrites": RewriteConfig(**rewrites)} if rewrites else {}))
    sim = fm.Simulation(atoms, field, seed=1, integrator=fm.integrators.Langevin(
        timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002),
        **({"engine": EngineConfig(**engine_options)} if engine_options else {}))
    engine = sim._engine
    sim.run(steps=2)
    torch.cuda.synchronize()
    edges = int(engine.force_field.edge_count)
    padded = max(engine.edge_sizes_used) if engine.edge_sizes_used else edges
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
    out = engine.force_field(engine.state.positions)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    del out
    trace = [event for device in snapshot["device_traces"] for event in device]
    peak, live = live_at_peak(trace)
    sites = defaultdict(lambda: [0, 0, set(), None])
    for size, frames in live.values():
        site, above = _site(frames)
        entry = sites[site]
        entry[0] += size
        entry[1] += 1
        entry[3] = entry[3] or above
    natoms = len(atoms)
    rows = sorted(([site, total, count, above] for site, (total, count, _, above) in sites.items()),
                  key=lambda row: -row[1])
    record = dict(tool="activation_profile", structure=name, precision=args.precision, natoms=natoms,
                  edges_at_cutoff=edges, padded_edges=padded, engine=engine_options, rewrites=rewrites,
                  allocated_before=base, peak_over_before=peak, plan=sim.plan.describe(),
                  sites=[dict(site=s, bytes=b, blocks=c, per_edge=b / padded, per_atom=b / natoms, above=a)
                         for s, b, c, a in rows])
    args.output.mkdir(parents=True, exist_ok=True)
    overrides = {**engine_options, **rewrites}
    tag = "".join(f"-{k}={v}" for k, v in overrides.items())
    path = args.output / f"profile-{name}-{args.precision}-{natoms}{tag}.json"
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(f"{name} {args.precision} {natoms} atoms, {padded} padded edges: peak {peak / 2**30:.2f} GiB above "
          f"{base / 2**30:.2f} GiB held before the call ({path})")
    for s, b, c, a in rows[:args.top]:
        print(f"{b / 2**30:7.3f} GiB {b / padded:8.0f} B/edge {b / natoms:9.0f} B/atom {c:5d} blocks  {s}")
    sim.close()
    os._exit(0)


if __name__ == "__main__":
    main()
