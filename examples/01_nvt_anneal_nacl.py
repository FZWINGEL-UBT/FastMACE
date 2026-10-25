"""NVT simulated anneal of NaCl with MACE MH-1 through ``fm.Simulation``.

The schedule is the anneal profile of the Langevin experiment: 0 -> 1000 K in 20 ps, then -3 K/ps to 250 K
(270 ps, 270,000 steps at 1 fs). ``--steps`` runs only the first steps of it, so a smoke run is cheap; run it
again with ``--steps 0`` for the whole schedule.

    python examples/01_nvt_anneal_nacl.py --structure NaCl.cif --checkpoint ~/models/mace-mh-1.model --steps 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def main(argv=None):
    args = _common.parser(__doc__.splitlines()[0], name="01_nvt_anneal_nacl", steps=200,
                          structure="NaCl.cif").parse_args(argv)

    from ase.io import read

    import fastmace as fm  # before torch (soft file locks, SPEC §14)

    atoms = read(args.structure).repeat((3, 3, 3))  # 3x3x3 supercell, as in production
    model = _common.load_mh1(args.checkpoint)
    ff = fm.ForceField(model, atoms, precision="float32")
    sim = fm.Simulation(
        atoms, ff,
        integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=0.0, friction_per_fs=0.002),
        workload=fm.Workload(max_temperature_K=1000),
        output=fm.Output(args.output, trajectory_every=10, thermo_every=10, restart_every=5000),
        seed=args.seed,
    )
    print(sim.plan)
    anneal = fm.protocols.ramp(0, 1000, ps=20) + fm.protocols.ramp(1000, 250, rate_K_per_ps=-3)
    sim.run(schedule=anneal, steps=args.steps or None)  # pause after --steps; resume with fm.Simulation.resume
    sim.flush_output()
    print(sim.report.text())
    sim.close()
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
