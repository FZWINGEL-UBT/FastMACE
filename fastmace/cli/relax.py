"""``fastmace relax STRUCTURE``: relax one structure to ``--fmax`` (SPEC §7.5, §10.3; plan task 4-T09).

The optimizer is one of ``fastmace.protocols.profiles.OPTIMIZERS``: ``gpu-LBFGS`` and ``gpu-FIRE``
(``fm.optimize`` on the graph engine), ``ase-LBFGS`` and ``ase-BFGS`` (ASE's on ``fm.calc``'s
calculator), with bare ``LBFGS`` and ``BFGS`` meaning ASE's. The default is
``profiles.DEFAULT_OPTIMIZER``. The cell is fixed unless ``--cell-filter frechet``: ASE's
``FrechetCellFilter``, or its device port for the GPU optimizers, at ``--pressure`` GPa.

Writes the relaxed structure (``--output``, default ``<structure stem>.relaxed.extxyz``), its summary
next to it (``.json``: optimizer, steps, converged, energies, fmax, and the cell with a filter) and
the optimizer's log (``.log``). A relaxation that does not reach ``--fmax`` within ``--steps`` writes
them too and exits 1.
"""

import json
import sys
from pathlib import Path

from fastmace.cli.md import DISPERSION_CHOICES, _dispersion, usage_error
from fastmace.protocols.profiles import DEFAULT_OPTIMIZER, OPTIMIZERS

HELP = "relax a structure with the GPU or ASE optimizers, at fixed cell or with a cell filter"

CELL_FILTERS = ("frechet",)
PRECISIONS = ("float32", "float64")
FMAX_EV_A, MAX_STEPS = 0.01, 5000  # the protocols' relaxation (protocols.quench.QUENCH_MAX_STEPS)


def add_arguments(parser):
    parser.add_argument("structure", metavar="STRUCTURE", help="structure file (any format ASE reads)")
    model = parser.add_argument_group("model")
    model.add_argument("--model", required=True, help="model label (fastmace models), path or URL")
    model.add_argument("--head", help="head of a multi-head model (required for MH-1)")
    model.add_argument("--charge", type=int, help="total charge, for models that take it")
    model.add_argument("--spin-multiplicity", type=int, help="spin multiplicity, for models that take it")
    model.add_argument("--sha256", help="expected SHA-256 of a model file outside the label table")
    model.add_argument("--trusted", action="store_true", help="trust a model file without a known SHA-256 (it is "
                       "unpickled)")
    model.add_argument("--precision", choices=PRECISIONS, default="float32", help="force-field precision "
                       "(default float32)")
    model.add_argument("--dispersion", choices=DISPERSION_CHOICES,
                       help="D3(BJ) dispersion: expanded from the head's training functional (SPEC §7.7)")
    model.add_argument("--d3-functional", metavar="NAME", help="the D3 parameter set, with --dispersion d3bj")
    relax = parser.add_argument_group("relaxation")
    relax.add_argument("--optimizer", choices=OPTIMIZERS, default=DEFAULT_OPTIMIZER,
                       help=f"the optimizer (default {DEFAULT_OPTIMIZER}; LBFGS and BFGS are ASE's)")
    relax.add_argument("--fmax", type=float, default=FMAX_EV_A, metavar="EV_A",
                       help=f"converged when every force is below this, eV/Å (default {FMAX_EV_A})")
    relax.add_argument("--steps", type=int, default=MAX_STEPS, help=f"at most this many steps (default {MAX_STEPS})")
    relax.add_argument("--repeat", type=int, nargs=3, metavar=("NA", "NB", "NC"), help="supercell repeats (default "
                       "1 1 1)")
    relax.add_argument("--cell-filter", choices=CELL_FILTERS, help="relax the cell too (ASE's FrechetCellFilter)")
    relax.add_argument("--pressure", type=float, metavar="GPA", help="the cell filter's scalar pressure in GPa "
                       "(default 0)")
    relax.add_argument("--output", metavar="FILE", help="the relaxed structure (default <stem>.relaxed.extxyz); "
                       "its .json summary and .log go next to it")


def _checked(args):
    """``(dispersion, cell filter record)``; ``ValueError`` for a flag combination that is not a relaxation."""
    if not args.fmax > 0:
        raise ValueError(f"--fmax must be positive (eV/Å), got {args.fmax}")
    if args.steps < 1:
        raise ValueError(f"--steps must be a positive number of steps, got {args.steps}")
    if args.pressure is not None and args.cell_filter is None:
        raise ValueError("--pressure needs --cell-filter frechet (at fixed cell there is no pressure term)")
    if args.repeat is not None and any(n < 1 for n in args.repeat):
        raise ValueError(f"--repeat takes three positive ints, got {args.repeat}")
    dispersion = _dispersion(args)
    cell = None
    if args.cell_filter == "frechet":
        from fastmace.optimize import FrechetCell

        cell = FrechetCell(scalar_pressure_GPa=0.0 if args.pressure is None else args.pressure)
    return dispersion, cell


def run(args):
    try:
        dispersion, cell = _checked(args)
    except ValueError as error:
        return usage_error("relax", error)
    from ase.io import read

    from fastmace.model.loading import load_pristine
    from fastmace.protocols import quench

    try:
        atoms = read(args.structure)
    except (OSError, ValueError) as error:
        return usage_error("relax", f"cannot read {args.structure}: {error}")
    if args.repeat is not None:
        atoms = atoms.repeat(tuple(args.repeat))
    output = Path(args.output) if args.output is not None else Path(f"{Path(args.structure).stem}.relaxed.extxyz")
    model = load_pristine(args.model, args.head, sha256=args.sha256, trusted=args.trusted, charge=args.charge,
                          spin_multiplicity=args.spin_multiplicity)
    relaxer = quench.relaxer(args.optimizer, model, precision=args.precision, dispersion=dispersion,
                             cell_filter=cell)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        relaxed, summary = relaxer(atoms, args.fmax, args.steps, logfile=output.with_suffix(".log"))
    finally:
        relaxer.close()
    relaxed.write(output)
    record = {"structure": str(args.structure), "repeat": list(args.repeat or (1, 1, 1)), "atoms": len(relaxed),
              "model": model.label or str(model.path), "head": model.head, "precision": args.precision,
              "output": str(output), **summary}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cell_text = "at fixed cell" if cell is None else f"with the Frechet cell filter at {cell.scalar_pressure_GPa:g} GPa"
    print(f"{summary['optimizer']} {cell_text}: {summary['initial_eV_atom']:.6f} -> {summary['energy_eV_atom']:.6f} "
          f"eV/atom in {summary['steps']} steps ({summary['seconds']:.1f} s), fmax {summary['fmax_eV_A']:.4f} eV/Å")
    print(f"Output: {output} (summary {output.with_suffix('.json')})")
    if not summary["converged"]:
        print(f"fastmace relax: the relaxation did not reach fmax {args.fmax} eV/Å in {args.steps} steps "
              f"({output.with_suffix('.log')})", file=sys.stderr)
        return 1
    return 0
