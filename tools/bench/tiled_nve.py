"""NVE drift of the tiled call (``TileConfig.state_dtype``; results/tiled-speed-20261108).

Velocity Verlet with the tiled force field as ``tiled_ab.py`` builds it (the default evaluation plan, float32,
species index, tile lists), the state on the host, aliased, under the wavefront; ``--state-dtype`` stores that state
in 16 bits. The integrator runs in float64 on the device. Velocities from a seeded Maxwell-Boltzmann draw at
``--temperature-K`` without the centre-of-mass motion. Every step records the potential, kinetic and total energy;
the record's drift is the slope of the total energy per atom from a least-squares line over the run (meV per atom
and ps), and its spread the RMS of the total energy about that line. Writes
``<output>/tiled-nve-<structure>-<atoms><tag>.json``. Run on an idle GPU (``tools/ci/local_gpu.py run --quiet``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiled_capacity  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="SrTiO3")
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--receivers", type=int, default=2048)
    p.add_argument("--mode", choices=("eager", "padded", "compiled", "auto"), default="auto")
    p.add_argument("--wavefront", choices=("auto", "on", "off"), default="on")
    p.add_argument("--state-dtype", choices=("bfloat16", "float16"), default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt-fs", type=float, default=1.0)
    p.add_argument("--temperature-K", type=float, default=600.0)
    p.add_argument("--rattle", type=float, default=0.05, help="Å, seeded, before the run (off the perfect lattice)")
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--tag", default="")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(argv)


def drift(times_ps, values):
    """(slope per ps, RMS about the least-squares line) of ``values`` against ``times_ps``."""
    import numpy as np

    t, v = np.asarray(times_ps), np.asarray(values)
    slope, intercept = np.polyfit(t, v, 1)
    return float(slope), float(np.sqrt(np.mean((v - (slope * t + intercept)) ** 2)))


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import numpy as np
    import torch
    from ase import units

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, args.rattle, 0.0)
    n = len(atoms)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"tiled-nve-{args.structure}-{n}{args.tag}.json"
    record = dict(structure=args.structure, repeat=list(repeat), atoms=n, model=args.model, receivers=args.receivers,
                  mode=args.mode, wavefront=args.wavefront, state_dtype=args.state_dtype, steps=args.steps,
                  dt_fs=args.dt_fs, temperature_K=args.temperature_K, rattle=args.rattle, seed=args.seed,
                  device=torch.cuda.get_device_name(), torch=torch.__version__,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"))

    user = fm.ForceField(fm.load(args.model), atoms, precision="float32")
    planner = minimal.default_planner(user)
    plan = planner.plan_evaluation(user, atoms)
    with tiled_capacity.no_global_list(True):
        field = planner.build_evaluation(plan, user, atoms, call_rows=args.receivers, release_node_attrs=True,
                                         release_whole_system=True, global_list=False)
        tiled = TiledForceField(field, TileConfig(
            args.receivers, None, mode=args.mode, neighbours="tiles", budget_receivers=(), state="host",
            alias_state=True, wavefront=args.wavefront, state_dtype=args.state_dtype, species="index"))
        x = field._prepare_one_off(atoms)
    device = x.device
    masses = torch.tensor(atoms.get_masses(), dtype=torch.float64, device=device).unsqueeze(1)
    rng = np.random.default_rng(args.seed)
    v = torch.tensor(rng.normal(size=(n, 3)), dtype=torch.float64, device=device)
    v *= torch.sqrt(args.temperature_K * units.kB / masses)
    v -= (masses * v).sum(0) / masses.sum()
    dt = args.dt_fs * units.fs

    def forces(positions):
        out = tiled(positions)
        return float(out.energy.reshape(-1)[0]), out.forces.detach().to(torch.float64)

    started = time.perf_counter()
    epot, f = forces(x)
    rows = []
    for step in range(args.steps + 1):
        ekin = float(0.5 * (masses * v * v).sum())
        rows.append((step * args.dt_fs / 1000.0, epot, ekin, epot + ekin))
        if step % 100 == 0:
            print(f"step {step} epot {epot:.6f} ekin {ekin:.6f} etot {epot + ekin:.6f} "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
        if step == args.steps:
            break
        v += 0.5 * dt * f / masses
        x = x + dt * v
        epot, f = forces(x)
        v += 0.5 * dt * f / masses
    times = [r[0] for r in rows]
    slope, spread = drift(times, [r[3] / n for r in rows])
    record.update(seconds=time.perf_counter() - started, rows=rows,
                  drift_meV_per_atom_ps=1e3 * slope, spread_meV_per_atom=1e3 * spread,
                  temperature_end_K=2 * rows[-1][2] / (3 * n * units.kB), last=dict(tiled.last))
    path.write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(dict(atoms=n, state_dtype=args.state_dtype, drift_meV_per_atom_ps=1e3 * slope,
                          spread_meV_per_atom=1e3 * spread, seconds=record["seconds"])))


if __name__ == "__main__":
    main()
