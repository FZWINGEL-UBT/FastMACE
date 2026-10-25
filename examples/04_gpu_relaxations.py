"""GPU structure relaxations with FIRE and LBFGS, at fixed cell and with a variable cell.

Four relaxations of one structure on FastMACE's graph engine (``fm.optimize``, ASE 3.27's algorithms and
defaults): LBFGS and FIRE at fixed cell, then both with ``FrechetCell`` (ASE's ``FrechetCellFilter``) at
``--pressure`` GPa. The structure is rattled first so there is something to relax. ``--steps`` caps every
relaxation.

    python examples/04_gpu_relaxations.py --structure NaCl.cif --checkpoint ~/models/mace-mh-1.model --steps 50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def main(argv=None):
    p = _common.parser(__doc__.splitlines()[0], name="04_gpu_relaxations", steps=500, structure="NaCl.cif")
    p.add_argument("--fmax", type=float, default=0.01, help="convergence, eV/Å")
    p.add_argument("--pressure", type=float, default=0.0, help="variable-cell target pressure, GPa")
    p.add_argument("--repeat", type=int, nargs=3, default=(2, 2, 2), metavar=("NA", "NB", "NC"), help="supercell")
    args = p.parse_args(argv)

    from ase.io import read

    import fastmace as fm

    start = read(args.structure).repeat(tuple(args.repeat))
    start.rattle(0.05, seed=args.seed)
    model = _common.load_mh1(args.checkpoint)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cell = fm.optimize.FrechetCell(scalar_pressure_GPa=args.pressure)
    runs = {
        "lbfgs_fixed": fm.optimize.LBFGS(),
        "fire_fixed": fm.optimize.FIRE(),
        "lbfgs_cell": fm.optimize.LBFGS(cell_filter=cell),
        "fire_cell": fm.optimize.FIRE(cell_filter=cell),
    }
    for name, optimizer in runs.items():
        atoms = start.copy()
        ff = fm.ForceField(model, atoms, precision="float32")
        with fm.Simulation(atoms, ff, optimizer=optimizer, seed=args.seed) as sim:
            relaxed, summary = sim.relax(fmax=args.fmax, steps=args.steps)
        relaxed.write(out / f"{name}.extxyz")
        volume = relaxed.get_volume() / start.get_volume()
        print(f"{name}: {summary.get('steps')} steps, converged {summary.get('converged')}, "
              f"V/V0 {volume:.4f}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
