"""Tiled capacity path: memory falsifier and accuracy check (docs/design/tiled-backend.md, steps 2-3).

``point``: one supercell in this process. It builds the one-off evaluation plan of ``fm.ForceField`` for the
cell (the default rewrite set, float32), then the same plan eager (``compile_mode="none"``), and wraps the eager
force field in ``fastmace.model.tiled.TiledForceField`` with ``--receivers``. It records:

- the neighbour build (rebuild, compact) and its peak, apart from the force call;
- two tiled force calls (the first one warms the kernels up), each with its peak allocated memory above the
  memory allocated before it, its time, and the global buffers it holds (h1 and its adjoint, forces);
- with ``--compare eager``, the whole-system eager call of the same force field (the same modules on one graph);
  with ``--compare compiled``, the default compiled evaluation plan (``fm.ForceField.evaluate``'s path) built in
  the same process after the tiled calls. Energy per atom, the largest force-component and stress-component
  differences, and the whole call's peak.

``--rattle A`` displaces every atom by a uniform random vector of up to A Angstrom per component (seed 0) and
``--strain S`` applies a random symmetric strain of up to S, so the comparison is off equilibrium.

Writes ``<output>/tiled-<name>-<atoms>-R<receivers><tag>.json``.
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
import capacity_scan  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--receivers", type=int, default=8192)
    p.add_argument("--edges", type=int, default=None)
    p.add_argument("--compare", choices=("none", "eager", "compiled"), default="none")
    p.add_argument("--rattle", type=float, default=0.0)
    p.add_argument("--strain", type=float, default=0.0)
    p.add_argument("--calls", type=int, default=2, help="tiled force calls (the first one warms up)")
    p.add_argument("--tag", default="")
    p.add_argument("--neighbours", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="NeighbourConfig overrides of the plan, e.g. gpu_neighbour_list=off (the host build: O(N) "
                        "against the GPU screen's O(N^2) at millions of atoms)")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(argv)


def supercell(structure, repeat, rattle, strain):
    import numpy as np

    _, cell = capacity_scan.base_cell(structure)
    atoms = cell.repeat(repeat)
    rng = np.random.default_rng(0)
    if strain:
        e = rng.uniform(-strain, strain, (3, 3))
        e = 0.5 * (e + e.T)
        atoms.set_cell(atoms.cell.array @ (np.eye(3) + e), scale_atoms=True)
    if rattle:
        atoms.positions = atoms.positions + rng.uniform(-rattle, rattle, atoms.positions.shape)
    atoms.wrap()
    return atoms


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = supercell(args.structure, repeat, args.rattle, args.strain)
    n = len(atoms)
    record = dict(structure=args.structure, repeat=list(repeat), atoms=n, receivers=args.receivers,
                  edges=args.edges, rattle=args.rattle, strain=args.strain, model=args.model,
                  device=torch.cuda.get_device_name(), torch=torch.__version__,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), phases={})
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"tiled-{args.structure}-{n}-R{args.receivers}{args.tag}.json"

    def save():
        path.write_text(json.dumps(record, indent=1, default=str))

    def mem(before=0):
        torch.cuda.synchronize()
        return dict(allocated=int(torch.cuda.memory_allocated()), reserved=int(torch.cuda.memory_reserved()),
                    peak_above_before=int(torch.cuda.max_memory_allocated() - before))

    try:
        record["phase"] = "build"
        started = time.perf_counter()
        user = fm.ForceField(fm.load(args.model), atoms, precision="float32")
        record["phases"]["user"] = mem()
        planner = minimal.default_planner(user)
        plan = planner.plan_evaluation(user, atoms)
        record["phases"]["plan"] = mem()
        eager = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                       dynamic_edges=False))
        if args.neighbours:
            overrides = {key: capacity_scan._value(value) for key, value in
                         (item.split("=", 1) for item in args.neighbours)}
            record["neighbours"] = overrides
            eager = dataclasses.replace(eager, neighbours=dataclasses.replace(eager.neighbours, **overrides))
        field = planner.build_evaluation(eager, user, atoms)
        record["rewrites"] = list(getattr(field, "applied", ()))
        record["plan_memory"] = list(plan.decision("memory")) if any(d[0] == "memory" for d in plan.decisions) else None
        tiled = TiledForceField(field, TileConfig(args.receivers, args.edges))
        record["phases"]["build"] = dict(seconds=time.perf_counter() - started, **mem())
        save()

        record["phase"] = "neighbours"
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        positions = field._prepare_one_off(atoms)
        record["edges_within_cutoff"] = int(field.neighbours.count)
        record["phases"]["neighbours"] = mem(before)
        save()

        calls = []
        out = None
        for call in range(args.calls):
            record["phase"] = f"tiled call {call}"
            del out
            out = None
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            out = tiled(positions)
            torch.cuda.synchronize()
            entry = dict(seconds=time.perf_counter() - started, **mem(before), **tiled.last)
            # ``allocated`` is after the call (the tiled field keeps its global state between calls); the call's
            # true peak is what was allocated before it plus its peak above that.
            entry.update(allocated_before=int(before), total_peak=int(before) + entry["peak_above_before"])
            calls.append(entry)
            record["calls"] = calls
            save()
        energy = float(out.energy.reshape(-1)[0])
        forces = out.forces.detach().double()
        stress = out.stress.detach().double() if out.stress is not None else None
        record["tiled"] = dict(energy_eV=energy, energy_per_atom_eV=energy / n,
                               max_abs_force=float(forces.abs().max()),
                               stress=None if stress is None else stress.cpu().tolist())
        save()

        if args.compare != "none":
            record["phase"] = f"compare {args.compare}"
            ref_field = field
            if args.compare == "compiled":
                ref_field = planner.build_evaluation(plan, user, atoms)
                record["reference_rewrites"] = list(getattr(ref_field, "applied", ()))
                positions = ref_field._prepare_one_off(atoms)
            runs = []
            for _ in range(2):
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                ref = ref_field(positions)
                torch.cuda.synchronize()
                runs.append(dict(seconds=time.perf_counter() - started, **mem(before)))
            r_energy = float(ref.energy.detach().reshape(-1)[0])
            r_forces = ref.forces.detach().double()
            r_stress = ref.stress.detach().double().reshape(3, 3) if ref.stress is not None else None
            record["reference"] = dict(
                kind=args.compare, calls=runs, energy_eV=r_energy,
                energy_diff_meV_per_atom=1e3 * (energy - r_energy) / n,
                max_force_diff=float((forces - r_forces).abs().max()),
                rms_force_diff=float((forces - r_forces).pow(2).mean().sqrt()),
                max_abs_force=float(r_forces.abs().max()),
                max_stress_diff=None if stress is None else float((stress - r_stress).abs().max()),
                stress=None if r_stress is None else r_stress.cpu().tolist())
        # A list without pairs (or zero forces) is a failed build, not a capacity point (31 October 2026: the
        # pair screen silently did not launch at 1.73M and 2.0M atoms).
        valid, detail = capacity_scan.check_point(args.structure, float(field.neighbours.cutoff), n,
                                                  int(record.get("edges_within_cutoff") or 0),
                                                  record["tiled"]["max_abs_force"])
        record["valid"], record["check"] = valid, detail
        record["phase"] = "done" if record["valid"] else "invalid"
    except torch.cuda.OutOfMemoryError as error:
        record["error"] = f"OutOfMemoryError: {str(error).splitlines()[0]}"
        record["oom_at"] = record["phase"]
        import traceback

        record["traceback"] = [f"{frame.filename}:{frame.lineno} {frame.name}: {frame.line}"
                               for frame in traceback.extract_tb(error.__traceback__)][-25:]
    save()
    print(json.dumps({k: record.get(k) for k in ("atoms", "phase", "error", "calls", "tiled", "reference")},
                     default=str)[:3000])


if __name__ == "__main__":
    main()
