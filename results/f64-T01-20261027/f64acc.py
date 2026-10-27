"""Float64 accuracy probe: forces and energy of one rattled cell from a planner-built float64 engine.

Usage: python f64acc.py LABEL STRUCTURE BACKEND [FIELD=VALUE ...]  -> f64acc/<STRUCTURE>-<LABEL>.npz
The cuBLAS emulation is set by the environment (CUBLAS_EMULATE_DOUBLE_PRECISION ...), so one process per side.
"""

import sys
from pathlib import Path

import fastmace as fm  # before torch: soft file locks

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig  # noqa: E402
from fastmace.planner import fixed, minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

label, spec, backend = sys.argv[1:4]
rewrites = dict(item.split("=", 1) for item in sys.argv[4:])
if backend == "hybrid":
    ab_step_time.prepare_fastmace_hybrid()
name, atoms = ab_step_time.read_structure(spec, None)
atoms.positions += np.random.default_rng(20261027).uniform(-0.05, 0.05, atoms.positions.shape)
planner = minimal.MinimalPlanner(engine=EngineConfig(backend=backend))
hardware = planner.hardware()
field = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64",
                      rewrites=RewriteConfig(**rewrites) if rewrites else "auto")
plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0), hardware)
engine = planner.build(plan, field, atoms.copy(), temperature_K=300.0)
for entry in getattr(engine.force_field, "applied", ()):
    if entry.name == "radial_tables":
        print("radial_tables", entry.detail, getattr(entry.result, "passed", None),
              getattr(entry.result, "max_relative_error", None), getattr(entry.result, "reason", None), flush=True)
        for _, interaction in enumerate(getattr(engine.force_field.model, "interactions", ())):
            table = getattr(interaction, "radial_table", None)
            if table is not None:
                print("certificate", table.coef.dtype, table.fit_error, table.certificate.get("worst"), flush=True)
forces = []
for _ in range(2):
    out = engine.force_field(engine.state.positions)
    forces.append(out)
energy, f = forces[-1][0], forces[-1][1]
f = f.detach().double().cpu().numpy()
out = Path("f64acc")
out.mkdir(exist_ok=True)
np.savez(out / f"{spec}-{label}.npz", forces=f, energy=float(energy.detach().double().sum().cpu()),
         options=np.array(sorted(fixed.enabled_options(plan))))
print(label, spec, backend, "energy", float(energy.detach().double().sum().cpu()), "max|F|", np.abs(f).max(),
      "options", sorted(fixed.enabled_options(plan)), flush=True)
