"""Saved-tensor inventory of one MD force evaluation (plans/capacity.md CAP-T08: which tensors the backward keeps).

Builds the default plan's ``fm.Simulation`` for one supercell, as ``activation_profile.py`` does, runs two steps,
then calls the engine's force field once (forward and the force backward) with

- ``torch.autograd.graph.saved_tensors_hooks``: every tensor an autograd node saves for the backward, deduplicated by
  its backing storage, with its dtype, shape, the frame inside the model's packages that saved it and the
  ``grad_fn`` that produced it; a storage saved by several nodes counts once;
- the caching allocator's history (``activation_profile.live_at_peak``): which saved storages are alive at the
  moment of the call's largest live total, and what else is.

Each storage gets a shape class from its leading dimensions: ``per-edge`` (the padded or the actual edge count),
``per-atom`` (a multiple of the atom count: irrep-major node features) or ``fixed`` (weights, tables). Prints the
largest saved storages and the totals per class, and writes ``<output>/saved-<name>-<precision>-<atoms><tag>.json``.
``--engine compile_mode=none`` keeps the model eager, so the frames name the modules (compiled graphs save through
``CompiledFunction``). Run inside a local GPU reservation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activation_profile  # noqa: E402
import capacity_scan  # noqa: E402

PACKAGES = activation_profile.PACKAGES


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


def _site(stack):
    """The innermost frame inside the model's packages (not this tool, nor torch's autograd and custom-op wrappers), and
    the next four."""
    def short(frame):
        name = frame.filename
        for package in PACKAGES:
            at = name.find("/" + package)
            if at >= 0:
                name = name[at + 1:]
                break
        return f"{name}:{frame.lineno} {frame.name}"

    inner = [f for f in reversed(stack) if any("/" + p in f.filename for p in PACKAGES)
             and "tools/bench" not in f.filename and "torch/autograd" not in f.filename
             and "torch/_library" not in f.filename]
    if not inner:
        return "unattributed", []
    return short(inner[0]), [short(f) for f in inner[1:5]]


def shape_class(shape, edges, natoms):
    if not shape:
        return "fixed"
    lead = shape[0]
    if lead in edges:
        return "per-edge"
    if lead and lead % natoms == 0:  # also [k * atoms, C]: irrep-major node features (k = 2l + 1)
        return "per-atom"
    return "fixed"


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import importlib.util

    import torch

    import fastmace as fm
    from fastmace.config import EngineConfig, RewriteConfig

    name, cell = capacity_scan.base_cell(args.structure)
    repeat = tuple(int(n) for n in args.repeat.lower().split("x"))
    atoms = cell.repeat(repeat)
    atoms.set_pbc(True)
    spec = importlib.util.spec_from_file_location("_ab_frontends", activation_profile.TOOLS / "ab_frontends.py")
    frontends = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontends)
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
    natoms = len(atoms)
    actual = int(engine.force_field.edge_count)
    padded = max(engine.edge_sizes_used) if engine.edge_sizes_used else actual
    edges = {actual, padded, *engine.edge_sizes_used}

    saved = {}

    def pack(tensor):
        if tensor.device.type != "cuda":
            return tensor
        storage = tensor.untyped_storage()
        key = storage.data_ptr()
        entry = saved.get(key)
        if entry is None:
            site, above = _site(traceback.extract_stack()[:-1])
            grad_fn = tensor.grad_fn.name() if tensor.grad_fn is not None else ("parameter" if tensor.requires_grad
                                                                                else "leaf")
            entry = saved[key] = dict(bytes=storage.nbytes(), dtype=str(tensor.dtype).replace("torch.", ""),
                                      shape=list(tensor.shape), site=site, above=above, grad_fn=grad_fn, saves=0)
        entry["saves"] += 1
        return tensor

    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = engine.force_field(engine.state.positions)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    del out
    trace = [event for device in snapshot["device_traces"] for event in device]
    peak, live = activation_profile.live_at_peak(trace)
    blocks = sorted((addr, addr + size) for addr, (size, _) in live.items())

    def alive(ptr):
        for start, stop in blocks:
            if start <= ptr < stop:
                return True
        return False

    rows = []
    for ptr, entry in saved.items():
        entry["class"] = shape_class(entry["shape"], edges, natoms)
        entry["alive_at_peak"] = alive(ptr)
        unit = padded if entry["class"] == "per-edge" else natoms if entry["class"] == "per-atom" else 1
        entry["bytes_per_unit"] = entry["bytes"] / unit
        rows.append(entry)
    rows.sort(key=lambda row: -row["bytes"])
    totals = {}
    for row in rows:
        key = (row["class"], row["alive_at_peak"])
        totals[key] = totals.get(key, 0) + row["bytes"]
    live_total = sum(size for size, _ in live.values())
    record = dict(tool="saved_tensors", structure=name, precision=args.precision, natoms=natoms,
                  edges_at_cutoff=actual, padded_edges=padded, engine=engine_options, rewrites=rewrites,
                  allocated_before=base, peak_over_before=peak, live_at_peak=live_total,
                  saved_totals={f"{c}{'' if a else ' (freed before the peak)'}": b for (c, a), b in totals.items()},
                  saved=rows, plan=sim.plan.describe())
    args.output.mkdir(parents=True, exist_ok=True)
    tag = "".join(f"-{k}={v}" for k, v in {**engine_options, **rewrites}.items())
    path = args.output / f"saved-{name}-{args.precision}-{natoms}{tag}.json"
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(f"{name} {args.precision} {natoms} atoms, {padded} padded edges ({actual} at the cutoff): peak "
          f"{peak / 2**30:.2f} GiB above {base / 2**30:.2f} GiB; {len(rows)} saved storages ({path})")
    for (cls, at_peak), total in sorted(totals.items(), key=lambda item: -item[1]):
        unit = padded if cls == "per-edge" else natoms if cls == "per-atom" else None
        per = f", {total / unit:,.0f} B per {'edge' if cls == 'per-edge' else 'atom'}" if unit else ""
        print(f"  saved {cls}{'' if at_peak else ' (freed before the peak)'}: {total / 2**30:.3f} GiB{per}")
    for row in rows[:args.top]:
        print(f"{row['bytes'] / 2**30:7.3f} GiB {row['class']:8s} {row['bytes_per_unit']:9.0f} B/unit "
              f"{'peak' if row['alive_at_peak'] else '    '} {row['dtype']:8s} {str(row['shape']):22s} "
              f"{row['grad_fn'][:24]:24s} {row['site']}")
    sim.close()
    sys.stdout.flush()  # os._exit skips the buffers of a redirected stdout
    os._exit(0)


if __name__ == "__main__":
    main()
