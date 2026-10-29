"""Which cells the default plan turns ``recompute_conv_weights`` on for (plans/capacity.md CAP-T07).

Plans each supercell with the minimal planner, as ``fm.Simulation`` does for Langevin NVT at 300 K, on this GPU's
free memory; nothing is loaded onto the GPU or compiled. Prints, per cell, the plan's value of
``RewriteConfig.recompute_conv_weights``, the reason and rule 12's predicted peak, and writes ``<output>``
(JSON). Run inside a local GPU reservation, so that the free memory is the idle GPU's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity_scan  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("cells", nargs="+", metavar="STRUCTURE@NXxNYxNZ")
    args = p.parse_args(argv)
    import fastmace as fm
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    fields = {}  # one ForceField per structure: its species superset
    rows = []
    for cell in args.cells:
        structure, _, repeat = cell.partition("@")
        name, base = capacity_scan.base_cell(structure)
        atoms = base.repeat(tuple(int(n) for n in repeat.lower().split("x")))
        atoms.set_pbc(True)
        if name not in fields:
            fields[name] = fm.ForceField(fm.load(args.model), atoms, precision=args.precision)
        field = fields[name]
        integrator = fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002)
        plan = minimal.default_planner(field).plan(field, atoms, integrator.to_config(seed=1),
                                                   Workload(max_temperature_K=300.0))
        _, choice, reason = plan.decision("RewriteConfig.recompute_conv_weights")
        memory = [str(r) for key, _, r in plan.decisions if key == "memory"]
        rows.append(dict(cell=f"{name}@{repeat}", natoms=len(atoms), recompute_conv_weights=choice,
                         reason=str(reason), memory=memory))
        print(f"{name}@{repeat} {len(atoms)} atoms: recompute_conv_weights={choice}: {reason}")
        for line in memory:
            print(f"    {line}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
