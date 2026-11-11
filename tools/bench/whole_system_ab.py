"""Same-process A/B of module constants read when the whole-system force field is built (decision 66).

Sides "a" and "b" differ only in module constants (``--a`` / ``--b MODULE:NAME=VALUE ...``, as
``tools/ab_rewrites.py``), set while that side's force field is built (the one-off evaluation plan's force field,
float32, as ``tools/virial_ab.py`` builds it) and its first calls compile; e.g.
``fastmace.model.forces:WHOLE_SYSTEM_FLOAT64_EDGES=True``. Both sides are built in the order given, warmed, then
timed in alternating blocks (``--blocks`` per side of ``--calls`` calls on fixed positions); the result is the
ratio of the sides' median block times, b/a. The record also compares b's energy, forces and stress with a's.
The call is the evaluation plan's (energy, forces and stress: an NPT step's or a relaxation's).
Run on an idle GPU (``tools/ci/local_gpu.py run --quiet``). Writes ``<output>/whole-ab-<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tiled_capacity  # noqa: E402
from ab_rewrites import constant_overrides  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="SrTiO3")
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--offset", type=float, default=0.0, help="Angstrom added to every coordinate")
    p.add_argument("--a", nargs="*", default=[], metavar="MODULE:NAME=VALUE")
    p.add_argument("--b", nargs="*", default=[], metavar="MODULE:NAME=VALUE")
    p.add_argument("--order", choices=("ab", "ba"), default="ab")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--tag", default="")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(argv)


def apply(overrides):
    saved = []
    for (module, name), value in overrides.items():
        saved.append((module, name, getattr(sys.modules[module], name)))
        setattr(sys.modules[module], name, value)
    return saved


def restore(saved):
    for module, name, value in reversed(saved):
        setattr(sys.modules[module], name, value)


def build(fm, model, atoms, overrides, warmup):
    import torch

    saved = apply(overrides)
    try:
        user = fm.ForceField(model, atoms, precision="float32")
        field = user._evaluator(atoms)[1]
        positions = field._prepare_one_off(atoms)
        def call():
            return field(positions)

        for _ in range(warmup):  # compiles with the constants bound
            out = call()
        torch.cuda.synchronize()
    finally:
        restore(saved)
    return dict(user=user, field=field, call=call, energy=float(out.energy.detach().double().sum()),
                forces=out.forces.detach().double().cpu(),
                stress=None if out.stress is None else out.stress.detach().double().cpu(),
                float64_edges=bool(getattr(field, "float64_edges", False)))


def block(side, calls):
    import torch

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(calls):
        side["call"]()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / calls


def main(argv=None):
    args = parse_args(argv)
    import torch

    import fastmace as fm

    nx, ny, nz = (int(x) for x in args.repeat.split("x"))
    atoms = tiled_capacity.supercell(args.structure, (nx, ny, nz), 0.0, 0.0)
    atoms.positions = atoms.positions + args.offset
    model = fm.load(args.model)
    overrides = {"a": constant_overrides(args.a), "b": constant_overrides(args.b)}
    sides = {name: build(fm, model, atoms, overrides[name], args.warmup) for name in args.order}
    times = {"a": [], "b": []}
    for _ in range(args.blocks):
        for name in args.order:
            times[name].append(block(sides[name], args.calls))
    a, b = sides["a"], sides["b"]
    ma, mb = statistics.median(times["a"]), statistics.median(times["b"])
    rec = dict(gpu=torch.cuda.get_device_name(), atoms=len(atoms), repeat=args.repeat, offset=args.offset,
               order=args.order, a=args.a, b=args.b, calls=args.calls,
               float64_edges={k: v["float64_edges"] for k, v in sides.items()},
               times_s=times, a_s=ma, b_s=mb, b_over_a=mb / ma,
               us_per_atom={"a": ma * 1e6 / len(atoms), "b": mb * 1e6 / len(atoms)},
               energy={"a": a["energy"], "b": b["energy"]},
               forces_max_abs=float((b["forces"] - a["forces"]).abs().max()),
               forces_max=float(a["forces"].abs().max()),
               peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    if a["stress"] is not None and b["stress"] is not None:
        rec["stress_max_abs"] = float((b["stress"] - a["stress"]).abs().max())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"whole-ab{args.tag}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "times_s"}))


if __name__ == "__main__":
    main()
