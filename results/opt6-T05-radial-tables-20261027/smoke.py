"""OPT6-T05 smoke: MH-1 float32 default plan against radial_tables=on, both against float64, one rattled cell each.

Prints the records' certificates (the applied detail of radial_tables in a planner-built engine), the force
differences of each float32 side from the float64 plan, and their difference from each other.
Usage: python smoke.py NaCl NbCl5 NaNbO3
"""

import json
import sys
from pathlib import Path

import fastmace as fm  # before torch: soft file locks

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig  # noqa: E402
from fastmace.planner import minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

ab_step_time.prepare_fastmace_hybrid()
model = fm.load("mh1-r2scan")
for spec in sys.argv[1:]:
    name, atoms = ab_step_time.read_structure(spec, None)
    atoms.positions += np.random.default_rng(20261027).uniform(-0.05, 0.05, atoms.positions.shape)
    # The certificate, from a planner-built engine with its self-checks.
    field = fm.ForceField(model, atoms, precision="float32", rewrites=RewriteConfig(radial_tables="on"))
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0),
                        planner.hardware())
    engine = planner.build(plan, field, atoms.copy(), temperature_K=300.0)
    applied = [dict(name=entry.name, detail=entry.detail, passed=getattr(entry.result, "passed", None),
                    error=getattr(entry.result, "max_relative_error", None))
               for entry in engine.force_field.applied if entry.name == "radial_tables"]
    del engine, field
    torch.cuda.empty_cache()
    forces = {}
    for side, precision, rewrites in (("float64", "float64", "auto"), ("default", "float32", RewriteConfig()),
                                      ("tables", "float32", RewriteConfig(radial_tables="on"))):
        ff = fm.ForceField(model, atoms, precision=precision, rewrites=rewrites)
        ff.evaluate(atoms)
        energy, f, _ = ff.evaluate(atoms)
        forces[side] = (float(energy), np.asarray(f, dtype=np.float64))
        del ff
        torch.cuda.empty_cache()
    e64, f64 = forces["float64"]
    report = dict(structure=str(name), natoms=len(atoms), applied=applied, max_force=float(np.abs(f64).max()))
    for side in ("default", "tables"):
        energy, f = forces[side]
        report[side] = dict(max_force_error=float(np.abs(f - f64).max()),
                            rms_force_error=float(np.sqrt(((f - f64) ** 2).mean())),
                            energy_error_meV_atom=1000 * (energy - e64) / len(atoms))
    report["tables_minus_default_max_force"] = float(np.abs(forces["tables"][1] - forces["default"][1]).max())
    print(json.dumps(report, default=str), flush=True)
