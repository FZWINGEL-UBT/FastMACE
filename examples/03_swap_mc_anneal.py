"""Swap MC anneal: FastMACE's shipped ``swap_anneal_NaNbOCl4_short`` protocol (O <-> Cl swaps on NaNbOCl4).

``fastmace run swap_anneal_NaNbOCl4_short`` runs the shipped TOML
(``fastmace/protocols/data/swap_anneal_NaNbOCl4_short.toml``). This script runs the same protocol from Python
(``tests/cpu/test_examples.py`` checks that ``protocol()`` equals the shipped file): an LBFGS relaxation on the
GPU, then an NVT anneal 0 -> 1000 K in 2 ps and -30 K/ps to 250 K with an O <-> Cl swap block (50 attempts
every 100 steps) wherever the target is at or above 250 K, then a quench of picked frames. MC moves are
experimental until their qualification study is recorded. ``--steps`` runs only the first MD steps (the swaps
start at 250 K, after 250 steps at 1 fs), then writes a restart; continue with ``fastmace run --resume OUTPUT``.

    python examples/03_swap_mc_anneal.py --checkpoint ~/models/mace-mh-1.model --steps 600 --repeat 1 1 1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

PROTOCOL = "swap_anneal_NaNbOCl4_short"


def protocol():
    """``(structure, fm.md options, stages)`` of the shipped protocol, as ``fastmace run`` reads them."""
    import fastmace as fm
    from fastmace.config import EngineConfig
    from fastmace.protocols import runner

    options = dict(
        model="mh1", head="matpes_r2scan", repeat=(3, 3, 3), timestep_fs=1.0, min_mc_temperature_K=250.0,
        schedule=fm.protocols.ramp(0.0, 1000.0, ps=2.0) + fm.protocols.ramp(1000.0, 250.0, rate_K_per_ps=-30.0),
        engine=EngineConfig(backend="hybrid"),
        output=fm.Output("swap_anneal", trajectory_every=10, thermo_every=10, restart_every=10_000),
        mc=(fm.mc.Swap(pairs=[("O", "Cl")], every=100, attempts=50, velocities="exchange"),),
    )
    stages = {"relax": {"fmax": 0.01, "optimizer": "gpu-LBFGS", "max_steps": 5000},
              "quench": {"min_K": 500.0, "bin_K": 100.0, "per_bin": 2, "fmax": 0.01, "optimizer": "gpu-LBFGS",
                         "max_steps": 5000}}
    return str(runner.DATA / "NaNbOCl4_I4m_Ma2026.cif"), options, stages


def main(argv=None):
    p = _common.parser(__doc__.splitlines()[0], name="03_swap_mc_anneal", steps=600)
    p.add_argument("--repeat", type=int, nargs=3, default=None, metavar=("NA", "NB", "NC"),
                   help="override the protocol's supercell (3 3 3, 756 atoms)")
    args = p.parse_args(argv)

    import dataclasses

    from fastmace.protocols import runner
    from fastmace.protocols.profiles import _quench

    structure, options, stages = protocol()
    options["output"] = dataclasses.replace(options["output"], directory=args.output)
    options["seed"] = args.seed
    if args.repeat is not None:
        options["repeat"] = tuple(args.repeat)
    options["model"] = _common.load_mh1(args.checkpoint, options.pop("head"))
    stages = {"relax": runner._relax(stages["relax"]), "quench": _quench(stages["quench"])}
    record = runner.start(structure, options, steps=args.steps, source=PROTOCOL, **stages)
    print(f"status: {record['status']}; output: {args.output}")


if __name__ == "__main__":
    main()
