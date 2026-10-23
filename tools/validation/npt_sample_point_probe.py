"""3b-T05: does the GPU production engine sample T, PE and V at the same point of the step as ASE? (GPU)

The comparison's FastMACE side samples through ``fm.Simulation``'s output sinks on the default planner's plan
(pipelined, captured graphs). This probe runs that path and ASE's ``LangevinBAOAB`` (the comparison tool's
``run_ase``) from identical velocities and noise streams, on a tiny MH-1-class model with D3, and compares every
sample: kinetic temperature (3N), PE and V.

- ``float64``: the plan in float64; ASE's MACE calculator in float64. The rows must agree to rounding, so the
  sample point, the kinetic energy and the noise stream are ASE's.
- ``float32``: the production precision; the rows agree only as far as float32 forces allow over the run.

    python tools/validation/npt_sample_point_probe.py --output OUT.json [--steps 200]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _comparison():
    spec = importlib.util.spec_from_file_location("_npt_d3_comparison", ROOT / "tools/validation/npt_d3_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "tests")]

    import fastmace as fm  # before torch: soft file locks, SPEC §14

    # isort: split
    import torch
    from ase import units
    from ase.build import bulk
    from ase.calculators.mixing import SumCalculator
    from ase.md.langevinbaoab import LangevinBAOAB
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.model import loading
    from fastmace.reference.d3 import torch_dftd3_calculator

    tool = _comparison()
    temperature, seed, steps = 300.0, 4, args.steps
    start = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    start.rattle(0.05, seed=3)
    MaxwellBoltzmannDistribution(start, temperature_K=temperature, rng=np.random.default_rng(99))
    directory = Path(tempfile.mkdtemp(prefix="fastmace-probe-"))
    path = directory / "tiny-mh1-11-17.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)

    rows = {}
    plans = {}
    for precision in ("float64", "float32"):
        atoms = start.copy()
        model = fm.load(str(path), trusted=True)
        force_field = fm.ForceField(model, atoms, precision=precision, dispersion=tool.d3_record())
        integrator = fm.integrators.LangevinBAOAB(
            timestep_fs=1.0, temperature_K=temperature, friction_per_fs=0.02 * units.fs, pressure_GPa=0.0,
            cell_mode="hydrostatic", barostat_tau_fs=2000.0, barostat_mass_factor=1.0)
        simulation = fm.Simulation(atoms, force_field, integrator=integrator, seed=seed,
                                   workload=fm.Workload(max_temperature_K=temperature))
        collected = []

        def frame(step, values, collected=collected, natoms=len(atoms)):
            host = {k: np.asarray(v, dtype=float) for k, v in values.items() if k in ("cell", "kinetic", "energy",
                                                                                      "stress")}
            collected.append(tool.observables_row(step, 1.0, natoms, host["cell"], float(host["kinetic"].sum()),
                                                  float(host["energy"].reshape(-1)[0]), host["stress"]))

        simulation.submit_frame(frame)
        simulation.add_sink(frame, 1)
        simulation.run(steps)
        simulation.flush_output()
        plans[precision] = json.loads(json.dumps(simulation.metadata().get("plan", {}).get("engine", {}), default=str))
        rows[f"fastmace_{precision}"] = np.array(collected)
        del simulation, force_field

    # ASE last: MACECalculator sets torch's global default dtype.
    loaded = loading.load_pristine(str(path), trusted=True)
    atoms = start.copy()
    atoms.calc = SumCalculator([loaded.calculator, torch_dftd3_calculator(tool.d3_record(), device="cuda")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dyn = LangevinBAOAB(atoms, timestep=1.0 * units.fs, temperature_K=temperature, externalstress=0.0,
                            hydrostatic=True, T_tau=1.0 / 0.02, P_tau=2000.0 * units.fs, P_mass_factor=1.0,
                            rng=np.random.default_rng(seed))
    collected = []
    for step in range(steps + 1):
        if step:
            dyn.run(1)
        collected.append(tool.sample_row(step, 1.0, atoms, atoms.get_potential_energy(),
                                         atoms.get_stress(voigt=False)))
    ase = np.array(collected)

    names = {"T_K": tool.COLUMNS.index("temperature_K"), "PE_eV_atom": tool.COLUMNS.index("potential_eV_atom"),
             "V_A3_atom": tool.COLUMNS.index("volume_A3_atom")}
    out = {"steps": steps, "plans": plans, "max_abs_difference": {}, "mean_T_K": {"ase": float(ase[1:, 2].mean())}}
    for side, data in rows.items():
        out["max_abs_difference"][side] = {name: float(np.abs(data[:, i] - ase[:, i]).max())
                                          for name, i in names.items()}
        out["mean_T_K"][side] = float(data[1:, names["T_K"]].mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("max_abs_difference", "mean_T_K")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
