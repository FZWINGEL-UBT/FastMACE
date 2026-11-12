"""Stress-call A/B of ``rewrites.engine.VIRIAL_ELEMENTWISE`` (the one-graph strain as elementwise products, so the
strain gradient is a reduction, not cuBLAS's split-K [3, E] x [E, 3] GEMM).

NVE MD has no strain leaf, so ``tools/ab_rewrites.py`` cannot see the switch. This times the force call with stress
(the one-off evaluation plan's compiled force field, the call an NPT step or a cell relaxation makes) on fixed
positions: side a = the switch off, side b = on, both built in one process in the order given, then timed blocks in
the pattern given (abab or abba). Per side also: one profiled call (kernel time per atom, split-K GEMM kernels), and
the stress and forces of b against a.

With ``--a-constant`` / ``--b-constant MODULE:NAME=VALUE`` the sides instead differ by those module constants (set
around each side's build and first compiled calls, as ``tools/ab_rewrites.py`` does; VIRIAL_ELEMENTWISE then stays
at its default on both), and ``--b-expect MODULE:NAME`` names proof counters (an int, or a function returning one)
that must advance over side b's build (exit 3 otherwise): the stress-mode A/B of any build-time switch.

Usage: python tools/virial_ab.py --checkpoint PATH --structure CIF@NxNxN --output OUT.json [--order ab|ba]
       [--a-constant M:N=V ...] [--b-constant M:N=V ...] [--b-expect M:N ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def parse_constant(item):
    """MODULE:NAME=VALUE -> (module, name, value); VALUE True/False/None, an int, a float or a string."""
    target, sep, text = item.partition("=")
    module, colon, name = target.partition(":")
    if not (sep and colon and module and name):
        raise ValueError(f"{item!r}: expected MODULE:NAME=VALUE")
    literal = {"True": True, "False": False, "None": None}
    if text in literal:
        return module, name, literal[text]
    for kind in (int, float):
        try:
            return module, name, kind(text)
        except ValueError:
            pass
    return module, name, text


def counters(items):
    import importlib

    values = {item: getattr(importlib.import_module(item.split(":")[0]), item.split(":")[1]) for item in items}
    return {item: value() if callable(value) else value for item, value in values.items()}


def build(fm, model, atoms, value, constants=()):
    """One side: VIRIAL_ELEMENTWISE = ``value`` (None: leave it) and ``constants`` [(module, name, value)] set
    while the side builds and compiles, then restored."""
    import importlib

    import torch

    from fastmace.rewrites import engine

    saved = engine.VIRIAL_ELEMENTWISE
    if value is not None:
        engine.VIRIAL_ELEMENTWISE = value
    restore = []
    for module, name, new in constants:
        target = importlib.import_module(module)
        restore.append((target, name, getattr(target, name)))
        setattr(target, name, new)
    try:
        user = fm.ForceField(model, atoms, precision="float32")
        field = user._evaluator(atoms)[1]
        positions = field._prepare_one_off(atoms)
        for _ in range(3):  # compiles with the constant bound
            out = field(positions)
        torch.cuda.synchronize()
    finally:
        engine.VIRIAL_ELEMENTWISE = saved
        for target, name, old in reversed(restore):
            setattr(target, name, old)
    return dict(user=user, field=field, positions=positions, strain_path=getattr(field, "strain_path", None),
                stress=out.stress.detach().double().cpu(), forces=out.forces.detach().double().cpu())


def profiled(side, atoms):
    import torch
    from torch.profiler import ProfilerActivity, profile

    field, positions = side["field"], side["positions"]
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        field(positions)
        torch.cuda.synchronize()
    total, largek, largek_us = 0.0, 0, 0.0
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        total += e.device_time
        if "largek" in e.name or "splitK" in e.name:
            largek += 1
            largek_us += e.device_time
    return dict(kernel_ns_per_atom=round(total * 1e3 / len(atoms), 1), largek_kernels=largek,
                largek_ns_per_atom=round(largek_us * 1e3 / len(atoms), 2))


def block(side, calls):
    import torch

    field, positions = side["field"], side["positions"]
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(calls):
        field(positions)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / calls * 1e3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None, help="Default: the omat0-medium label's file")
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", required=True, help="CIF[@NxNxN]")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--order", choices=("ab", "ba"), default="ab")
    p.add_argument("--pattern", choices=("abab", "abba"), default="abab")
    p.add_argument("--blocks", type=int, default=6, help="Timed blocks per side")
    p.add_argument("--calls", type=int, default=20, help="Force calls per block")
    p.add_argument("--a-constant", nargs="*", default=[], metavar="MODULE:NAME=VALUE", help="Side a's module constants")
    p.add_argument("--b-constant", nargs="*", default=[], metavar="MODULE:NAME=VALUE", help="Side b's, as --a-constant")
    p.add_argument("--b-expect", nargs="*", default=[], metavar="MODULE:NAME",
                   help="Proof counters that must advance over side b's build (exit 3 otherwise)")
    args = p.parse_args()
    constants = {"a": [parse_constant(c) for c in args.a_constant], "b": [parse_constant(c) for c in args.b_constant]}
    switch = not (constants["a"] or constants["b"])  # no constants: the VIRIAL_ELEMENTWISE A/B

    import ase.io
    import torch

    import fastmace as fm

    path, _, rep = args.structure.partition("@")
    atoms = ase.io.read(path)
    if rep:
        atoms = atoms.repeat(tuple(int(x) for x in rep.split("x")))
    model = (fm.load(args.model, checkpoint=args.checkpoint) if args.checkpoint else fm.load(args.model))
    sides, advanced = {}, {}
    for name in args.order:
        before = counters(args.b_expect)
        sides[name] = build(fm, model, atoms, (name == "b") if switch else None, constants[name])
        advanced[name] = {item: value - before[item] for item, value in counters(args.b_expect).items()}
    failed = [item for item, value in advanced["b"].items() if value <= 0]
    rec = dict(gpu=torch.cuda.get_device_name(), atoms=len(atoms), order=args.order, pattern=args.pattern,
               calls=args.calls, constants={k: [f"{m}:{n}={v}" for m, n, v in c] for k, c in constants.items()},
               expect=advanced, expect_failed=failed,
               strain_path={k: str(v["strain_path"]) for k, v in sides.items()},
               profile={k: profiled(v, atoms) for k, v in sides.items()})
    a, b = sides["a"], sides["b"]
    rec["stress_max_abs"] = float((b["stress"] - a["stress"]).abs().max())
    rec["stress_max"] = float(a["stress"].abs().max())
    rec["forces_max_abs"] = float((b["forces"] - a["forces"]).abs().max())
    for name in "ab":
        block(sides[name], 2)  # warm both after the profiler
    times = {"a": [], "b": []}
    sequence = (("ab" if args.pattern == "abab" else "abba") * args.blocks)[: 2 * args.blocks]
    for name in sequence:
        times[name].append(block(sides[name], args.calls))
    ma, mb = statistics.median(times["a"]), statistics.median(times["b"])
    rec.update(times_ms=times, a_ms=round(ma, 4), b_ms=round(mb, 4), b_over_a=round(mb / ma, 4))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("gpu", "atoms", "order", "strain_path", "profile", "stress_max_abs",
                                         "stress_max", "forces_max_abs", "a_ms", "b_ms", "b_over_a", "expect")}))
    if failed:
        print(f"EXPECT FAILED: {failed} did not advance over side b's build")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
