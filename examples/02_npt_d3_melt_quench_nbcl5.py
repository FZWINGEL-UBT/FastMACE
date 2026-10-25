"""NPT + D3(BJ) short melt-quench of NbCl5 with MACE MH-1 through ``fm.md``.

The molecular chlorides need dispersion at constant pressure: without D3, NbCl5 swells by about 20 % at
300 K. The schedule is the short NPT protocol: 0 -> 1000 K in 2 ps, then -30 K/ps to 250 K (27 ps at 1 fs),
hydrostatic Langevin-BAOAB at 0 GPa. ``--steps`` runs only the first steps. Afterwards the last frame is
relaxed on the GPU (LBFGS with MACE + the same D3 record), the 0 K end of a melt-quench.

    python examples/02_npt_d3_melt_quench_nbcl5.py --structure NbCl5.cif --checkpoint mace-mh-1.model --steps 50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def main(argv=None):
    p = _common.parser(__doc__.splitlines()[0], name="02_npt_d3_melt_quench_nbcl5", steps=200,
                       structure="NbCl5.cif")
    p.add_argument("--repeat", type=int, nargs=3, default=(1, 1, 1), metavar=("NA", "NB", "NC"),
                   help="supercell (production uses 3 3 3, 1944 atoms)")
    p.add_argument("--relax-steps", type=int, default=200, help="LBFGS steps for the final quench")
    args = p.parse_args(argv)

    import fastmace as fm

    model = _common.load_mh1(args.checkpoint)
    melt_quench = fm.protocols.ramp(0, 1000, ps=2) + fm.protocols.ramp(1000, 250, rate_K_per_ps=-30)
    run = fm.md(args.structure, model=model, ensemble="NPT", pressure_GPa=0.0, cell_mode="hydrostatic",
                schedule=melt_quench, timestep_fs=1.0, steps=args.steps or None, dispersion="d3bj",
                repeat=tuple(args.repeat), seed=args.seed,
                output=fm.Output(args.output, trajectory_every=10, thermo_every=10, restart_every=5000))
    print(run.summary)

    # The quench: relax the last committed frame at its own (NPT) cell, MACE + the same D3 record.
    sim = run.simulation
    from ase.io import read

    atoms = read(args.structure).repeat(tuple(args.repeat))  # the run's structure, at its committed state:
    committed = sim.committed()
    atoms.set_cell(committed.cell, scale_atoms=False)
    atoms.set_positions(committed.positions)
    dispersion = sim.force_field.dispersion
    sim.close()
    ff = fm.ForceField(model, atoms, precision="float32", dispersion=dispersion)
    relax = fm.Simulation(atoms, ff, optimizer=fm.optimize.LBFGS(), seed=args.seed)
    relaxed, summary = relax.relax(fmax=0.01, steps=args.relax_steps)
    relax.close()
    out = Path(args.output) / "quenched.extxyz"
    relaxed.write(out)
    print(f"quench: {summary['steps']} steps, converged {summary['converged']}, "
          f"{summary['energy_eV_atom']:.6f} eV/atom, fmax {summary['fmax_eV_A']:.4f} eV/Å -> {out}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
