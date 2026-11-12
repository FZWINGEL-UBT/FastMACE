"""Force/energy consistency probes of one configuration (paper experiment 1; ``tools/run_manifest.py``'s a, b, c).

One process per configuration (the switch constants are set before anything is built). Every evaluation is
``fm.ForceField.evaluate`` (the one-off evaluation plan: compiled, the graph plan's rewrite set, energy accumulated
in float64), positions in float64 on the host.

  derivative  for each structure, ``--directions`` random unit displacement fields d (unit 2-norm over all 3N
              components) and steps h: the central difference (E(x + h d) - E(x - h d)) / 2h against -F(x).d
  knots       one atom moved along a straight line (``--line-A`` long, ``--line-step-A`` steps), atom and direction
              chosen among random candidates for the most knot crossings of its edges (the attached spline's
              knots, from configuration a's or b's build; c has none and reuses ``--knots-from``): E and F.u at
              every point; the largest step-to-step jump of E about a smooth (degree-8 Chebyshev) trend, and the
              work -sum F.dR (trapezoid) against E(end) - E(start)
  loop        the same atom around a closed equilateral triangle (``--loop-side-A``, ``--loop-steps``): the
              trapezoid sum of F.dR (zero to round-off and O(step^2) for conservative forces)

Structures: the force/stress gate's rattled SrTiO3 3x3x3 cell (135 atoms; the gate's first hydrostatic
configuration: seed 20261015, strain 0.02, rattle 0.05 A) and the same recipe on SrTiO3 10x10x10. Output:
``<output>/probe-<config>.json`` with the manifest (``run_manifest.manifest``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import run_manifest  # noqa: E402

GATE_SEED, GATE_STRAIN, GATE_RATTLE = 20261015, 0.02, 0.05


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", choices=sorted(run_manifest.CONFIGS), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--cif", type=Path, required=True, help="SrTiO3 unit cell (Benchmark_loup's CIF)")
    p.add_argument("--repeats", nargs="+", default=["3x3x3", "10x10x10"])
    p.add_argument("--directions", type=int, default=20)
    p.add_argument("--steps-A", nargs="+", type=float, default=[1e-3, 3e-4, 1e-4])
    p.add_argument("--line-A", type=float, default=0.05)
    p.add_argument("--line-step-A", type=float, default=1e-4)
    p.add_argument("--loop-side-A", type=float, default=0.02)
    p.add_argument("--loop-steps", type=int, default=600)
    p.add_argument("--candidates", type=int, default=400, help="(atom, direction) candidates for the line")
    p.add_argument("--knots-from", type=Path, default=None,
                   help="A probe JSON whose knots / atom / direction to reuse (same line in every configuration)")
    p.add_argument("--skip", nargs="*", default=[], choices=("derivative", "knots", "loop"))
    p.add_argument("--seed", type=int, default=12)
    return p.parse_args(argv)


def gate_configuration(cif, repeat):
    from ase.io import read

    spec = importlib.util.spec_from_file_location("_force_stress_gate", TOOLS / "force_stress_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    base = read(cif).repeat(tuple(int(n) for n in repeat.split("x")))
    base.set_pbc(True)
    return gate.make_configurations(base, "hydrostatic", 1, GATE_SEED, GATE_STRAIN, GATE_RATTLE)[0]


class Evaluator:
    def __init__(self, field, atoms):
        self.field, self.atoms, self.calls, self.seconds = field, atoms.copy(), 0, 0.0

    def __call__(self, positions):
        import numpy as np

        self.atoms.positions = positions
        started = time.perf_counter()
        energy, forces, _ = self.field.evaluate(self.atoms)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        return float(energy), np.asarray(forces, dtype=np.float64)


def derivative(evaluate, x0, directions, steps, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    e0, f0 = evaluate(x0)
    rows = []
    for k in range(directions):
        d = rng.standard_normal(x0.shape)
        d /= np.linalg.norm(d)
        slope = -float(np.sum(f0 * d))  # dE/dh predicted by the forces
        row = dict(direction=k, minus_F_dot_d=slope, central={})
        for h in steps:
            ep, _ = evaluate(x0 + h * d)
            em, _ = evaluate(x0 - h * d)
            cd = (ep - em) / (2 * h)
            row["central"][f"{h:g}"] = dict(cd=cd, mismatch=cd - slope, relative=(cd - slope) / abs(slope))
        rows.append(row)
    summary = {}
    for h in steps:
        rel = np.array([abs(r["central"][f"{h:g}"]["relative"]) for r in rows])
        mis = np.array([r["central"][f"{h:g}"]["mismatch"] for r in rows])
        summary[f"{h:g}"] = dict(median_abs_relative=float(np.median(rel)), max_abs_relative=float(rel.max()),
                                 rms_relative=float(np.sqrt(np.mean(rel ** 2))),
                                 mean_mismatch_eV_per_A=float(mis.mean()),
                                 rms_mismatch_eV_per_A=float(np.sqrt(np.mean(mis ** 2))))
    return dict(energy_eV=e0, force_rms_eV_per_A=float(np.sqrt(np.mean(f0 ** 2))), rows=rows, summary=summary)


def edge_crossings(positions, cell, atom, direction, length, knots, r_max):
    """Knot crossings of ``atom``'s edges (every periodic image within r_max + length) when it moves by
    length * direction, from 51 points along the line."""
    import numpy as np

    frac = (positions - positions[atom]) @ np.linalg.inv(cell)
    frac -= np.round(frac)
    ts = np.linspace(0.0, length, 51)
    count = 0
    for shift in [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]:
        vec = (frac + np.array(shift, dtype=float)) @ cell
        r0 = np.linalg.norm(vec, axis=1)
        for v in vec[(r0 > 1e-8) & (r0 < r_max + length)]:
            r = np.linalg.norm(v[None, :] - ts[:, None] * direction[None, :], axis=1)
            r = r[r < r_max]
            if r.size > 1:
                count += int(np.sum(np.abs(np.diff(np.searchsorted(knots, r)))))
    return count


def choose_line(x0, cell, knots, r_max, length, candidates, seed, natoms):
    import numpy as np

    rng = np.random.default_rng(seed + 1)
    best = None
    for _ in range(candidates):
        atom = int(rng.integers(natoms))
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        n = edge_crossings(x0, cell, atom, u, length, knots, r_max)
        if best is None or n > best[0]:
            best = (n, atom, u)
    return dict(crossings=best[0], atom=best[1], direction=best[2].tolist())


def smooth_residual(s, values, degree=8):
    import numpy as np

    fit = np.polynomial.chebyshev.Chebyshev.fit(s, values, degree)
    return values - fit(s)


def line(evaluate, x0, atom, u, length, step):
    import numpy as np

    n = int(round(length / step))
    s = np.arange(n + 1) * step
    energies, fdu = [], []
    for value in s:
        x = x0.copy()
        x[atom] += value * u
        e, f = evaluate(x)
        energies.append(e)
        fdu.append(float(f[atom] @ u))
    energies, fdu = np.array(energies), np.array(fdu)
    residual = smooth_residual(s, energies - energies[0])
    jumps = np.abs(np.diff(residual))
    work = -float(np.sum(0.5 * (fdu[1:] + fdu[:-1]) * step))  # -int F.dR
    delta = float(energies[-1] - energies[0])
    # the same for the forces' own integral: residual of the cumulative work about E (path consistency)
    cumulative = np.concatenate([[0.0], -np.cumsum(0.5 * (fdu[1:] + fdu[:-1]) * step)])
    gap = (energies - energies[0]) - cumulative
    return dict(points=int(n + 1), step_A=step, energies_eV=energies.tolist(), F_dot_u_eV_per_A=fdu.tolist(),
                max_jump_eV=float(jumps.max()), rms_jump_eV=float(np.sqrt(np.mean(jumps ** 2))),
                max_jump_index=int(jumps.argmax()), residual_rms_eV=float(np.sqrt(np.mean(residual ** 2))),
                delta_E_eV=delta, work_eV=work, delta_E_minus_work_eV=delta - work,
                max_abs_E_minus_cumulative_work_eV=float(np.abs(gap).max()))


def loop(evaluate, x0, atom, side, steps, seed):
    import numpy as np

    rng = np.random.default_rng(seed + 2)
    a = rng.standard_normal(3)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(3)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    vertices = [np.zeros(3), side * a, side * (0.5 * a + np.sqrt(3) / 2 * b)]
    per_side = steps // 3
    path = []
    for k in range(3):
        p, q = vertices[k], vertices[(k + 1) % 3]
        for j in range(per_side):
            path.append(p + (q - p) * j / per_side)
    path.append(vertices[0])
    path = np.array(path)
    forces, energies = [], []
    for point in path:
        x = x0.copy()
        x[atom] += point
        e, f = evaluate(x)
        energies.append(e)
        forces.append(f[atom].copy())
    forces = np.array(forces)
    dr = np.diff(path, axis=0)
    work = float(np.sum(0.5 * (forces[1:] + forces[:-1]) * dr))  # closed-loop integral of F.dR
    seg = 0.5 * np.sum((forces[1:] + forces[:-1]) * dr, axis=1)
    return dict(side_A=side, steps=len(dr), plane=[a.tolist(), b.tolist()], loop_work_eV=work,
                sum_abs_segment_work_eV=float(np.abs(seg).sum()), energy_closure_eV=float(energies[-1] - energies[0]),
                energies_eV=energies, F_dot_dR_eV=seg.tolist())


def main(argv=None):
    args = parse_args(argv)
    rewrites = run_manifest.apply_config(args.config)
    import numpy as np
    import torch

    import fastmace as fm
    from fastmace.config import RewriteConfig

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"probe-{args.config}.json"
    model = fm.load(args.model, checkpoint=args.checkpoint) if "checkpoint" in fm.load.__code__.co_varnames \
        else fm.load(args.model)
    record = dict(tool="consistency_probe", structures={})
    for repeat in args.repeats:
        atoms = gate_configuration(args.cif, repeat)
        field = fm.ForceField(model, atoms, precision="float32",
                              rewrites=RewriteConfig(**rewrites) if rewrites else "auto")
        evaluate = Evaluator(field, atoms)
        x0 = atoms.positions.astype(np.float64).copy()
        started = time.perf_counter()
        evaluate(x0)  # build and compile
        setup_s = time.perf_counter() - started
        plan = field.evaluation_plan(atoms)
        entry = dict(repeat=repeat, natoms=len(atoms), setup_s=setup_s,
                     manifest=run_manifest.manifest(config=args.config, checkpoint=args.checkpoint,
                                                    structure=f"{args.cif}@{repeat} gate hydrostatic-0",
                                                    atoms=atoms, plan=plan))
        if "derivative" not in args.skip:
            entry["derivative"] = derivative(evaluate, x0, args.directions, args.steps_A, args.seed)
            print(args.config, repeat, "derivative", json.dumps(entry["derivative"]["summary"]), flush=True)
        if repeat == args.repeats[0] and not {"knots", "loop"} <= set(args.skip):
            knots, spline = run_manifest.spline_knots()
            if args.knots_from is not None:
                ref = json.loads(args.knots_from.read_text())["structures"][repeat]["line_choice"]
                choice = dict(ref)
                if knots is None:
                    knots = np.array(ref["knots"])
            elif knots is not None:
                choice = choose_line(x0, atoms.cell.array, knots, float(spline["r_max"]), args.line_A,
                                     args.candidates, args.seed, len(atoms))
                choice.update(knots=knots.tolist(), r_max=float(spline["r_max"]))
            else:
                raise SystemExit("no spline attached: pass --knots-from a probe of configuration a or b")
            entry["line_choice"] = choice
            u = np.array(choice["direction"])
            if "knots" not in args.skip:
                entry["knots"] = line(evaluate, x0, choice["atom"], u, args.line_A, args.line_step_A)
                print(args.config, repeat, "knots", {k: v for k, v in entry["knots"].items()
                                                     if not isinstance(v, list)}, flush=True)
            if "loop" not in args.skip:
                entry["loop"] = loop(evaluate, x0, choice["atom"], args.loop_side_A, args.loop_steps, args.seed)
                print(args.config, repeat, "loop", {k: v for k, v in entry["loop"].items()
                                                    if not isinstance(v, list)}, flush=True)
        entry.update(evaluations=evaluate.calls, evaluate_s=evaluate.seconds)
        record["structures"][repeat] = entry
        out.write_text(json.dumps(record, indent=1, default=str) + "\n")
        del field, evaluate
        torch.cuda.empty_cache()
    print("wrote", out)


if __name__ == "__main__":
    main()
