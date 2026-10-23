"""3b-T05: where the PE offset between the comparison's engines comes from (GPU, one process).

The comparison's 300 K potential energies differ by about 0.8 meV/atom between FastMACE (energy accumulated
in float64, SPEC §7.3) and the ASE side (MACE's ``MACECalculator`` in float32, which sums the energy in float32;
1b-T03 measured that summation error). This probe evaluates the ASE runs' final frames (``final.extxyz`` of
``tools/validation/npt_d3_comparison.py run-ase``) three ways, each plus D3 on ``fm.D3("r2scan")``:

- ``ase_float32``: the ASE side's calculator (cuEq float32) plus torch-dftd (float64);
- ``ase_float64``: the same in float64, the reference;
- ``fastmace``: ``fm.ForceField(MH-1, precision="float32", dispersion=...)`` on the default planner's plan,
  ``evaluate``.

It prints the per-frame energies (eV/atom) and the differences in meV/atom.

    python tools/validation/npt_d3_energy_offset.py --runs DIR/ase-* --output OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _comparison():
    spec = importlib.util.spec_from_file_location("_npt_d3_comparison", HERE / "npt_d3_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="ASE run directories with final.extxyz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import fastmace as fm  # before torch: soft file locks, SPEC §14

    # isort: split
    from ase.calculators.mixing import SumCalculator
    from ase.io import read
    from mace.calculators import MACECalculator

    from fastmace.reference.d3 import torch_dftd3_calculator

    tool = _comparison()
    record = tool.d3_record()
    frames = [(run.name, read(run / "final.extxyz")) for run in args.runs]
    model = fm.load(str(args.checkpoint), head=tool.SETTINGS["head"], trusted=True)
    force_field = fm.ForceField(model, frames[0][1], precision="float32", dispersion=record)
    results = {"frames": {}}
    # FastMACE first, for every frame: MACECalculator sets torch's global default dtype, which a compiled
    # FastMACE call made afterwards would trace with (the first version of this probe failed so).
    fastmace_energies = {name: force_field.evaluate(atoms)[0] / len(atoms) for name, atoms in frames}
    for name, atoms in frames:
        natoms = len(atoms)
        energies = {"fastmace": fastmace_energies[name]}
        for dtype in ("float32", "float64"):
            mace = MACECalculator(model_paths=str(args.checkpoint), device="cuda", default_dtype=dtype,
                                  head=tool.SETTINGS["head"], enable_cueq=True)
            calc = SumCalculator([mace, torch_dftd3_calculator(record, dtype="float64", device="cuda")])
            probe = atoms.copy()
            probe.calc = calc
            energies[f"ase_{dtype}"] = float(probe.get_potential_energy()) / natoms
        energies["meV_atom"] = {
            "fastmace_minus_float64": 1e3 * (energies["fastmace"] - energies["ase_float64"]),
            "ase_float32_minus_float64": 1e3 * (energies["ase_float32"] - energies["ase_float64"]),
            "fastmace_minus_ase_float32": 1e3 * (energies["fastmace"] - energies["ase_float32"]),
        }
        results["frames"][name] = energies
        print(name, json.dumps(energies["meV_atom"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
