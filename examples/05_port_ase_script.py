"""Porting an ASE script to FastMACE: ``fm.calc`` for single points, ``fastmace.ase.Langevin`` for MD.

The ASE original would be

    atoms.calc = MACECalculator(model_paths=..., head="matpes_r2scan")
    dyn = ase.md.langevin.Langevin(atoms, 1 * units.fs, temperature_K=300, friction=0.002 / units.fs)
    dyn.attach(log, interval=10); dyn.run(steps)

The port changes two lines: the calculator becomes ``fm.calc`` (compiled tier, ASE drives every call, so the
speed-up is the smaller one) and the integrator comes from ``fastmace.ase`` with ``model=`` (FastMACE owns
the MD loop on its graph engine: the full speed-up). ASE's arguments, units and observers stay.

    python examples/05_port_ase_script.py --structure NaCl.cif --checkpoint ~/models/mace-mh-1.model --steps 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def main(argv=None):
    p = _common.parser(__doc__.splitlines()[0], name="05_port_ase_script", steps=100, structure="NaCl.cif")
    p.add_argument("--repeat", type=int, nargs=3, default=(3, 3, 3), metavar=("NA", "NB", "NC"), help="supercell")
    args = p.parse_args(argv)

    from ase import units
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    import fastmace as fm
    from fastmace.ase import Langevin  # was: from ase.md.langevin import Langevin

    model = _common.load_mh1(args.checkpoint)
    atoms = read(args.structure).repeat(tuple(args.repeat))

    # 1. Single points through the ASE calculator interface (was: MACECalculator).
    atoms.calc = fm.calc(model)
    print(f"E = {atoms.get_potential_energy():.6f} eV, max |F| = {abs(atoms.get_forces()).max():.4f} eV/Å")

    # 2. MD: same ASE arguments; model= hands the loop to FastMACE.
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    atoms.calc = None
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=__import__("numpy").random.default_rng(args.seed))
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.002 / units.fs, model=model,
                   seed=args.seed, trajectory=str(out / "md.traj"), logfile=str(out / "md.log"), loginterval=10,
                   output=str(out / "md.fastmace"))

    def report():
        print(f"step {dyn.get_number_of_steps():6d}  T = {atoms.get_temperature():7.1f} K")

    dyn.attach(report, interval=max(1, args.steps // 5))
    dyn.run(args.steps)
    dyn.close()
    print(f"output: {out}")


if __name__ == "__main__":
    main()
