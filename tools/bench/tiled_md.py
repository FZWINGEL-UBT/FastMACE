"""Tiled capacity path: a short reproducible NVE check with neighbour-list rebuilds (docs/design/tiled-backend.md).

Velocity Verlet in float64 on the host side of the force field, with the force field's own skin list: every step
compacts the edges within the cutoff, and rebuilds the candidates when an atom moved more than half the skin. The
forces come from ``--path tiled`` (``TiledForceField`` with ``--receivers``), ``whole`` (the same eager force
field's whole-system call) or ``both``: two runs from the same start (Maxwell-Boltzmann velocities at
``--temperature``, seed 0, zero total momentum), compared step by step (positions, potential energy) and by their
total-energy drift.

Writes ``<output>/md-<name>-<atoms>-R<receivers>.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiled_capacity  # noqa: E402

KB_EV = 8.617333262e-5
# ASE units: 1 fs = 0.09822694788464063 (eV, Å, amu)
FS = 0.09822694788464063


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--repeat", required=True)
    p.add_argument("--receivers", type=int, default=4096)
    p.add_argument("--mode", choices=("eager", "padded", "graphs", "compiled"), default="eager",
                   help="TileConfig.mode; 'compiled' builds the compiled evaluation plan (its whole-system call is "
                        "then the compiled model's)")
    p.add_argument("--path", choices=("tiled", "whole", "both"), default="both")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--timestep-fs", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=600.0)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(argv)


def run(field, call, atoms, steps, dt_fs, temperature):
    import numpy as np
    import torch

    device = field.device
    masses = torch.tensor(atoms.get_masses(), dtype=torch.float64, device=device)[:, None]
    rng = np.random.default_rng(0)
    v = rng.standard_normal((len(atoms), 3)) * np.sqrt(KB_EV * temperature / atoms.get_masses())[:, None]
    v -= (atoms.get_masses()[:, None] * v).sum(0) / atoms.get_masses().sum()
    v = torch.tensor(v, dtype=torch.float64, device=device)
    x = field._prepare_one_off(atoms).clone()
    dt = dt_fs * FS
    out = call(x)
    f = out.forces.detach().double()
    rows, rebuilds = [], 0
    started = time.perf_counter()
    for step in range(steps + 1):
        kinetic = float(0.5 * (masses * v * v).sum())
        rows.append(dict(step=step, epot=float(out.energy.reshape(-1)[0]), ekin=kinetic,
                         etot=float(out.energy.reshape(-1)[0]) + kinetic, edges=int(field.edge_count),
                         rebuilds=rebuilds))
        if step == steps:
            break
        v = v + 0.5 * dt * f / masses
        x = x + dt * v
        if bool(field.needs_rebuild(x)):
            field.rebuild(x)
            rebuilds += 1
        field.compact(x)
        field.use_edges(int(field.edge_count))
        out = call(x)
        f = out.forces.detach().double()
        v = v + 0.5 * dt * f / masses
        if step in (0, 9, 49, 99, 199, 499):
            rows[-1]["positions_sample"] = x[:4].cpu().tolist()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return dict(rows=rows, seconds=seconds, rebuilds=rebuilds, final_positions=x, natoms=len(atoms))


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import numpy as np
    import torch

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    user = fm.ForceField(fm.load(args.model), atoms, precision="float32")
    planner = minimal.default_planner(user)
    plan = planner.plan_evaluation(user, atoms)
    if args.mode != "compiled":
        plan = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                      dynamic_edges=False))
    field = planner.build_evaluation(plan, user, atoms)
    tiled = TiledForceField(field, TileConfig(args.receivers, mode=args.mode))
    record = dict(structure=args.structure, repeat=list(repeat), atoms=len(atoms), receivers=args.receivers,
                  mode=args.mode,
                  steps=args.steps, timestep_fs=args.timestep_fs, temperature_K=args.temperature,
                  device=torch.cuda.get_device_name(), source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"),
                  skin=float(getattr(plan.neighbours, "skin", float("nan"))), runs={})
    paths = ("tiled", "whole") if args.path == "both" else (args.path,)
    finals = {}
    for path in paths:
        call = tiled if path == "tiled" else field
        result = run(field, call, atoms, args.steps, args.timestep_fs, args.temperature)
        finals[path] = result.pop("final_positions")
        etot = np.array([row["etot"] for row in result["rows"]])
        result["drift_meV_per_atom"] = float(1e3 * (etot[-1] - etot[0]) / len(atoms))
        result["etot_spread_meV_per_atom"] = float(1e3 * (etot.max() - etot.min()) / len(atoms))
        result["seconds_per_step"] = result["seconds"] / args.steps
        if path == "tiled":
            result["last_call"] = {k: v for k, v in tiled.last.items() if k != "slots"}
            result["slots"] = {str(k): list(v) for k, v in tiled.last.get("slots", {}).items()}
        record["runs"][path] = result
    if len(paths) == 2:
        a, b = record["runs"]["tiled"]["rows"], record["runs"]["whole"]["rows"]
        record["compare"] = dict(
            max_epot_diff_meV_per_atom=float(max(abs(p["epot"] - q["epot"]) for p, q in zip(a, b)) * 1e3 / len(atoms)),
            final_max_position_diff_A=float((finals["tiled"] - finals["whole"]).abs().max()),
            same_rebuild_count=record["runs"]["tiled"]["rebuilds"] == record["runs"]["whole"]["rebuilds"])
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"md-{args.structure}-{len(atoms)}-R{args.receivers}.json"
    out.write_text(json.dumps(record, indent=1))
    summary = {k: v for k, v in record.items() if k != "runs"}
    summary["runs"] = {p: {k: v for k, v in r.items() if k != "rows"} for p, r in record["runs"].items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
