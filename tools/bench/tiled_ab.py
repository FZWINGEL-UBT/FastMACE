"""Same-process A/B of module constants on the tiled call (results/tiled-speed-20261108; decision 66).

One supercell, one force field and one ``TiledForceField`` (as ``tiled_capacity.py`` builds them: the default
evaluation plan, float32, compiled tiles with ``--mode auto``). Sides "a" and "b" differ only in module constants
(``--a`` / ``--b MODULE:NAME=VALUE ...``, as ``tools/ab_rewrites.py``), set before each of the side's calls and
read by the call (or by the compiled sweeps, whose guards keep one specialisation per side once both have run).

Each side first runs ``--warmup`` calls (the first, eager call and the compile happen there), side a first with
``--order ab`` and side b first with ``ba``. Then ``--blocks`` blocks of ``--calls`` timed calls per side alternate in
that order; the result is the ratio of the sides' median call times, b/a (below 1: b is faster), with each block
pair's ratio. The last call of each side keeps its energy, forces and stress, and the record compares them (b
against a). Writes ``<output>/tiled-ab-<tag>.json``. Run on an idle GPU (``tools/ci/local_gpu.py run --quiet``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
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
    p.add_argument("--receivers", type=int, default=8192)
    p.add_argument("--no-budget", action="store_true", help="the receiver cap stays --receivers")
    p.add_argument("--mode", choices=("eager", "padded", "compiled", "auto"), default="auto")
    p.add_argument("--host-state", action="store_true")
    p.add_argument("--alias-state", action="store_true")
    p.add_argument("--wavefront", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--a", nargs="*", default=[], metavar="MODULE:NAME=VALUE")
    p.add_argument("--b", nargs="*", default=[], metavar="MODULE:NAME=VALUE")
    p.add_argument("--order", choices=("ab", "ba"), default="ab")
    p.add_argument("--warmup", type=int, default=3, help="untimed calls per side before the blocks")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--calls", type=int, default=3, help="timed calls per side and block")
    p.add_argument("--tag", default="")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(argv)


@contextlib.contextmanager
def constants(values):
    saved = {}
    try:
        for (module, name), value in values.items():
            saved[(module, name)] = getattr(sys.modules[module], name)
            setattr(sys.modules[module], name, value)
        yield
    finally:
        for (module, name), value in saved.items():
            setattr(sys.modules[module], name, value)


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    sides = {"a": constant_overrides(args.a), "b": constant_overrides(args.b)}
    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    record = dict(structure=args.structure, repeat=list(repeat), atoms=n, model=args.model, order=args.order,
                  a=args.a, b=args.b, receivers=args.receivers, mode=args.mode, host_state=args.host_state,
                  alias_state=args.alias_state, wavefront=args.wavefront, device=torch.cuda.get_device_name(),
                  torch=torch.__version__, source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"tiled-ab-{args.structure}-{n}{args.tag}-{args.order}.json"

    user = fm.ForceField(fm.load(args.model), atoms, precision="float32")
    planner = minimal.default_planner(user)
    plan = planner.plan_evaluation(user, atoms)
    with tiled_capacity.no_global_list(True):
        field = planner.build_evaluation(plan, user, atoms, call_rows=args.receivers, release_node_attrs=True,
                                         release_whole_system=True, global_list=False)
        tiled = TiledForceField(field, TileConfig(
            args.receivers, None, mode=args.mode, neighbours="tiles", budget_receivers=() if args.no_budget else None,
            state="host" if args.host_state else "device", alias_state=args.alias_state, wavefront=args.wavefront,
            species="index"))
        positions = field._prepare_one_off(atoms)
    record["rewrites"] = [str(r) for r in getattr(field, "applied", ())]

    kept = {}

    def call(side, keep=False):
        with constants(sides[side]):
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = tiled(positions)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
        if keep:
            kept[side] = dict(energy=float(out.energy.reshape(-1)[0]), forces=out.forces.detach().clone(),
                              stress=None if out.stress is None else out.stress.detach().double().cpu())
        del out
        return seconds

    order = list(args.order)
    record["warmup"] = {side: [call(side) for _ in range(args.warmup)] for side in order}
    record["last"] = {side: None for side in order}
    blocks = []
    for block in range(args.blocks):
        entry = {}
        for side in order:
            entry[side] = [call(side) for _ in range(args.calls)]
            record["last"][side] = dict(tiled.last)
        blocks.append(entry)
        record["blocks"] = blocks
        path.write_text(json.dumps(record, indent=1, default=str))
    for side in order:
        call(side, keep=True)
    medians = {side: statistics.median([s for entry in blocks for s in entry[side]]) for side in order}
    pairs = [statistics.median(entry["b"]) / statistics.median(entry["a"]) for entry in blocks]
    a, b = kept["a"], kept["b"]
    record["result"] = dict(
        median_s=medians, us_per_atom={side: 1e6 * medians[side] / n for side in order},
        ratio_b_over_a=medians["b"] / medians["a"], pair_ratios=pairs,
        energy_eV=dict(a=a["energy"], b=b["energy"]), energy_diff_meV_per_atom=1e3 * (b["energy"] - a["energy"]) / n,
        forces_bitwise=bool(torch.equal(a["forces"], b["forces"])),
        max_force_diff=float((a["forces"].double() - b["forces"].double()).abs().max()),
        max_abs_force=float(a["forces"].abs().max()),
        max_stress_diff=None if a["stress"] is None else float((a["stress"] - b["stress"]).abs().max()),
        stress_bitwise=None if a["stress"] is None else bool(torch.equal(a["stress"], b["stress"])))
    path.write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(dict(atoms=n, order=args.order, **record["result"]), default=str))


if __name__ == "__main__":
    main()
