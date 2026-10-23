"""GPU LBFGS on annealing quench frames against ASE's LBFGS on ``fm.calc`` (plan task 4-T05; SPEC §10.3).

For each structure, frames from the research engine's ``captured.traj`` (the frames its 13 October quench
relaxed with ASE's LBFGS) are relaxed

- by ``DeviceLBFGS`` on the planned MH-1 engine, per ``--modes`` (``graph``: captured, padded edges, rebuilds
  between replays; ``compiled``: the compiled tier, exact edge counts, no graphs) and ``--skins`` (the
  neighbour-list skin, Angstrom): conductor's choice 6, the mode and skin of optimiser plans;
- and by ASE 3.27's ``LBFGS`` on ``fm.calc`` (the compiled tier, the same model, head and precision), what
  ``protocols.quench.relax`` runs today. Only the optimiser differs.

Each relaxation records steps, converged, wall seconds, the start and final energies (eV/atom), fmax, the
engine's neighbour rebuilds and re-captures; the GPU minima are compared with ASE's (|dE| meV/atom and the
largest per-atom distance under the minimum image convention). LBFGS fixed cell, fmax 0.01 eV/A, memory
100, maxstep 0.2 A, at most 5,000 steps unless given. Hybrid float32 by default; run on an otherwise idle GPU
(``tools/ci/local_gpu.py run``) with the OEq environment's Python:

    PYTHONPATH=$PWD ~/environments/mace_cueq_oeq/bin/python tools/validation/lbfgs_frames.py \\
        --checkpoint ~/models/mace-mh-1.model --output results/4-T05-<machine>-<date>/frames \\
        --traj NaCl=<dir>/NaCl/captured.traj NbCl5=<dir>/NbCl5/captured.traj --frames 0 10 20

Writes ``<output>.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model", default="mh1")
    p.add_argument("--head", default="matpes_r2scan")
    p.add_argument("--traj", nargs="+", required=True, help="NAME=PATH of a captured.traj")
    p.add_argument("--frames", nargs="+", type=int, default=[0, 10, 20])
    p.add_argument("--modes", nargs="+", choices=("graph", "compiled"), default=["graph", "compiled"])
    p.add_argument("--skins", nargs="+", type=float, default=[0.3, 0.6, 1.0])
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--no-ase", action="store_true", help="Skip the ASE side")
    return p.parse_args(argv)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_engine(planner, force_field, atoms, mode):
    """The optimiser engine of an MD plan's force field (planner dispatch for optimisers is 4-T01's)."""
    import dataclasses

    from fastmace import optimize
    from fastmace.config import IntegratorConfig
    from fastmace.engine.compiled import CompiledEngine
    from fastmace.engine.graph import GraphEngine
    from fastmace.integrators.registry import build_dof, build_optimizer
    from fastmace.optimize.driver import OptimizerStepper
    from fastmace.planner.workload import Workload
    from fastmace.state import SimulationState

    plan = planner.plan(force_field, atoms, IntegratorConfig(), Workload(max_temperature_K=300.0),
                        planner.hardware(), mode=None if mode == "graph" else mode)
    md = planner.build(plan, force_field, atoms.copy(), temperature_K=0.0)
    field = md.force_field
    del md
    config = optimize.LBFGS().to_config()
    state = SimulationState.from_atoms(atoms, field.device, 0)
    stepper = OptimizerStepper(build_optimizer(build_dof(state, config), config))
    engine_config = dataclasses.replace(plan.engine, pipelined="off", speculative_graph=False)
    if plan.tier == "compiled":
        engine = CompiledEngine(engine_config, field, stepper, state, neighbours=plan.neighbours,
                                dynamic_edges=plan.engine.dynamic_edges)
    else:
        engine = GraphEngine(engine_config, field, stepper, state, neighbours=plan.neighbours)
        engine.prepare()
    return plan, engine


def mic_distance(a, b):
    """The largest per-atom distance between two configurations of one cell, minimum image."""
    import numpy as np

    delta = b.positions - a.positions
    fractional = np.linalg.solve(a.cell.array.T, delta.T).T
    fractional -= np.round(fractional)
    return float(np.linalg.norm(fractional @ a.cell.array, axis=1).max())


def ase_relax(calc, atoms, fmax, steps):
    import numpy as np
    from ase.optimize import LBFGS

    relaxed = atoms.copy()
    relaxed.set_momenta(np.zeros((len(relaxed), 3)))
    relaxed.calc = calc
    started = time.perf_counter()
    initial = float(relaxed.get_potential_energy())
    run = LBFGS(relaxed, logfile=None)
    converged = bool(run.run(fmax=fmax, steps=steps))
    seconds = time.perf_counter() - started
    energy = float(relaxed.get_potential_energy())
    out = {"converged": converged, "steps": int(run.get_number_of_steps()), "seconds": seconds,
           "initial_eV_atom": initial / len(relaxed), "energy_eV_atom": energy / len(relaxed),
           "fmax_eV_A": float(np.linalg.norm(relaxed.get_forces(), axis=1).max())}
    relaxed.calc = None
    return relaxed, out


def main(argv=None):
    args = parse_args(argv)
    import torch
    from ase.io import read

    from fastmace import config
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.optimize import driver
    from fastmace.planner.minimal import MinimalPlanner

    oeq = str(loading.prepare_hybrid()) if args.backend == "hybrid" else None
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    loaded = loading.load_pristine(args.model, args.head, checkpoint=str(checkpoint))
    meta = {"schema": 1, "task": "4-T05", "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fastmace_commit": git_commit(), "machine": __import__("socket").gethostname(),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "ase": __import__("ase").__version__,
            "backend": args.backend, "precision": args.precision, "oeq_runtime": oeq, "model": args.model,
            "head": args.head, "checkpoint": str(checkpoint), "fmax": args.fmax, "steps": args.steps,
            "optimizer": "LBFGS(maxstep_A=0.2, memory=100, damping=1.0, alpha=70.0)", "structures": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    calc = None
    if not args.no_ase:
        from fastmace.ase.calculator import FastMACECalculator

        calc = FastMACECalculator(loaded, precision=args.precision)
    for spec in args.traj:
        name, _, path = spec.partition("=")
        frames = {index: read(path, index=index) for index in args.frames}
        record = meta["structures"].setdefault(name, {"traj": path, "natoms": len(frames[args.frames[0]]),
                                                      "frames": {}})
        for index, atoms in frames.items():
            atoms.calc = None
            atoms.set_constraint()
            entry = record["frames"].setdefault(str(index), {"gpu": {}})
            reference = None
            if calc is not None:
                reference, entry["ase"] = ase_relax(calc, atoms, args.fmax, args.steps)
                print(f"{name} frame {index} ASE: {entry['ase']['steps']} steps {entry['ase']['seconds']:.1f} s "
                      f"E {entry['ase']['energy_eV_atom']:.6f}", flush=True)
                save()
            for mode in args.modes:
                for skin in args.skins:
                    key = f"{mode}-skin{skin:g}"
                    planner = MinimalPlanner(engine=config.EngineConfig(backend=args.backend),
                                             neighbours=config.NeighbourConfig(skin=skin))
                    force_field = ForceField(loaded, atoms, precision=args.precision)
                    started = time.perf_counter()
                    plan, engine = build_engine(planner, force_field, atoms, mode)
                    build_s = time.perf_counter() - started
                    summary = driver.relax(engine, fmax=args.fmax, steps=args.steps)
                    torch.cuda.synchronize()
                    result = {key: summary[key] for key in ("converged", "steps", "seconds", "initial_eV_atom",
                                                            "energy_eV_atom", "fmax_eV_A", "neighbour_rebuilds",
                                                            "recaptures")}
                    info = engine.info()
                    result.update(build_s=build_s, tier=plan.tier, skin_A=skin, edge_sizes=info["edge_sizes_used"],
                                  graphs=len(info["graphs"]), capture_count=len(info["recapture_log"]),
                                  ms_per_step=1000 * summary["seconds"] / max(summary["steps"], 1))
                    if reference is not None:
                        relaxed = atoms.copy()
                        relaxed.positions = engine.state.positions.detach().cpu().numpy()
                        result["dE_vs_ase_meV_atom"] = 1000 * (summary["energy_eV_atom"]
                                                               - entry["ase"]["energy_eV_atom"])
                        result["max_distance_vs_ase_A"] = mic_distance(reference, relaxed)
                    entry["gpu"][key] = result
                    print(f"{name} frame {index} {key}: {result['steps']} steps {result['seconds']:.1f} s "
                          f"({result['ms_per_step']:.2f} ms/step) rebuilds {result['neighbour_rebuilds']} "
                          f"recaptures {result['recaptures']} dE {result.get('dE_vs_ase_meV_atom', float('nan')):.4f} "
                          f"meV/atom dmax {result.get('max_distance_vs_ase_A', float('nan')):.4f} A", flush=True)
                    del engine, force_field
                    torch.cuda.empty_cache()
                    save()
    meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
