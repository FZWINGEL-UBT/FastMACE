"""Float64 package accuracy (decision 63): forces and stress of one rattled cell from a planner-built float64 engine.

Usage: python f64pkg_acc.py LABEL STRUCTURE [rewrite.FIELD=VALUE ...] [engine.FIELD=VALUE ...]
-> f64pkg/<STRUCTURE>-<LABEL>.npz. NPT plan (hydrostatic) so the force field computes the stress. The default
plan is the package (decision 63); the stock side passes every package feature off. One process per side: cuBLAS's
emulation setting is process-wide.
"""

import sys
from pathlib import Path

import fastmace as fm  # before torch: soft file locks

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402
import numpy as np  # noqa: E402

from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig  # noqa: E402
from fastmace.model import compile as compiling  # noqa: E402
from fastmace.planner import fixed, minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

label, spec = sys.argv[1:3]
options = [item.split("=", 1) for item in sys.argv[3:]]
rewrites = {k.split(".", 1)[1]: v for k, v in options if k.startswith("rewrite.")}
engine_options = {k.split(".", 1)[1]: v for k, v in options if k.startswith("engine.")}
ab_step_time.prepare_fastmace_hybrid()
name, atoms = ab_step_time.read_structure(spec, None)
atoms.positions += np.random.default_rng(20261027).uniform(-0.05, 0.05, atoms.positions.shape)
planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid", **engine_options))
hardware = planner.hardware()
field = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64",
                      rewrites=RewriteConfig(**rewrites) if rewrites else "auto")
integrator = IntegratorConfig(ensemble="npt", pressure_GPa=0.0, temperature_K=300.0)
plan = planner.plan(field, atoms, integrator, Workload(max_temperature_K=300.0), hardware)
engine = planner.build(plan, field, atoms.copy(), temperature_K=300.0)
out = engine.force_field(engine.state.positions)
out = engine.force_field(engine.state.positions)
energy, forces, stress = out[0], out[1], out[2]
f = forces.detach().double().cpu().numpy()
s = stress.detach().double().cpu().numpy()
Path("f64pkg").mkdir(exist_ok=True)
options_on = sorted(fixed.enabled_options(plan))
active = compiling.fp64_emulation_active(engine.state.positions.device)
np.savez(Path("f64pkg") / f"{spec}-{label}.npz", forces=f, stress=s, energy=float(energy.detach().double().sum().cpu()),
         options=np.array(options_on), emulation_active=active)
print(label, spec, "energy", float(energy.detach().double().sum().cpu()), "fp64_emulation", plan.engine.fp64_emulation,
      "active", active, "radial_tables", plan.rewrites.radial_tables, "species_tables", plan.rewrites.species_tables,
      flush=True)
