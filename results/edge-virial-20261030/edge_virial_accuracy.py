"""Edge-virial stress against the strain path, the accuracy part of the edge-virial falsifier.

The edge virial is MACE's own ``compute_edge_forces`` / ``get_atomic_virials_stresses`` (LAMMPS's per-atom virial):
one gradient of the energy by the edge vectors, ``sigma = sym(sum_e r_e (x) dE/dr_e) / V``, with no strain leaf.
The strain path is MACE's ``compute_stress`` (the displacement leaf). Both on the CPU with the e3nn modules, in
float32 and float64, on the force/stress gate's configurations (tools/force_stress_gate.py: random strain up to
``--strain``, then a Gaussian rattle). Reported per structure: max |sigma_a - sigma_b| over the configurations and
the nine components, in meV/A^3, for
  - edge64 - strain64 (the method: equal up to float64 rounding),
  - strain32 - strain64 (the float32 strain path's own error),
  - edge32 - strain64 (the float32 edge virial's error),
  - edge32 - strain32,
and the same for the forces (eV/A), which both paths give.
    python edge_virial_accuracy.py --checkpoint PATH --structure NaCl --count 20 --output DIR
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import numpy as np
import torch

from fastmace.model import loading

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def gate_module():
    spec = importlib.util.spec_from_file_location("_force_stress_gate", TOOLS / "force_stress_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(calc, atoms):
    """(strain-path stress, edge-virial stress, forces of each), as float64 numpy, ASE's sign (eV/A^3)."""
    model = calc.models[0]
    batch = calc._atoms_to_batch(atoms)
    volume = abs(np.linalg.det(atoms.cell.array))
    out = model(batch.to_dict(), training=False, compute_force=True, compute_stress=True)
    strain = out["stress"][0].detach().double().numpy()
    strain_forces = out["forces"].detach().double().numpy()
    batch = calc._atoms_to_batch(atoms)
    out = model(batch.to_dict(), training=False, compute_force=True, compute_stress=False, compute_edge_forces=True,
                compute_atomic_stresses=True)
    # MACE's atomic virials are -dE/d(strain) per atom (half of each edge to either end); the stress is
    # +dE/d(strain) / V.
    edge = -out["atomic_virials"].detach().double().sum(0).numpy() / volume
    edge_forces = out["forces"].detach().double().numpy()
    return strain, edge, strain_forces, edge_forces


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", required=True, help="NaCl, NbCl5, NaNbO3 (the gate's 3x3x3 cell) or NAME@AxBxC")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--strain", type=float, default=0.02)
    p.add_argument("--rattle-A", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    gate = gate_module()
    name, source, repeat = gate.structure_spec(args.structure, (3, 3, 3), os.environ.get("FASTMACE_STRUCTURES", "."))
    from ase.io import read
    base = read(source).repeat(repeat)
    base.set_pbc(True)
    configs = gate.make_configurations(base, "full", args.count, args.seed, args.strain, args.rattle_A)
    calcs = {dtype: loading.load(args.model, checkpoint=args.checkpoint, backend="e3nn", dtype=dtype,
                                 device="cpu").calculator for dtype in ("float64", "float32")}
    rows = []
    for index, atoms in enumerate(configs):
        started = time.perf_counter()
        r = {dtype: evaluate(calcs[dtype], atoms) for dtype in calcs}
        s64, e64, f64, g64 = r["float64"]
        s32, e32, f32, g32 = r["float32"]
        rows.append(dict(
            index=index, seconds=time.perf_counter() - started,
            stress_meV_A3={"edge64-strain64": 1e3 * float(np.abs(e64 - s64).max()),
                           "strain32-strain64": 1e3 * float(np.abs(s32 - s64).max()),
                           "edge32-strain64": 1e3 * float(np.abs(e32 - s64).max()),
                           "edge32-strain32": 1e3 * float(np.abs(e32 - s32).max())},
            forces_eV_A={"edge64-strain64": float(np.abs(g64 - f64).max()),
                         "strain32-strain64": float(np.abs(f32 - f64).max()),
                         "edge32-strain64": float(np.abs(g32 - f64).max())},
            stress64_meV_A3=(1e3 * s64).tolist()))
        print(f"{name} {index}: stress max|d| meV/A^3 {rows[-1]['stress_meV_A3']} ({rows[-1]['seconds']:.0f} s)",
              flush=True)
    summary = {key: max(row["stress_meV_A3"][key] for row in rows) for key in rows[0]["stress_meV_A3"]}
    forces = {key: max(row["forces_eV_A"][key] for row in rows) for key in rows[0]["forces_eV_A"]}
    print(f"{name} {len(base)} atoms, {len(rows)} configurations: max stress |d| meV/A^3 {summary}; "
          f"forces eV/A {forces}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{name}.json").write_text(json.dumps(dict(
        structure=name, atoms=len(base), count=args.count, strain=args.strain, rattle_A=args.rattle_A, seed=args.seed,
        torch=torch.__version__, threads=torch.get_num_threads(), host=os.uname().nodename,
        max_stress_meV_A3=summary, max_forces_eV_A=forces, rows=rows), indent=1))


if __name__ == "__main__":
    main()
