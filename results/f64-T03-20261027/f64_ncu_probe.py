"""f64-T03: one float64 package force call (default float64 plan, NbCl5 3x3x3) between cudaProfilerStart/Stop."""
import sys
from pathlib import Path

import fastmace as fm

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402
import torch  # noqa: E402

from fastmace.config import EngineConfig, IntegratorConfig  # noqa: E402
from fastmace.planner import minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

spec = sys.argv[1] if len(sys.argv) > 1 else "NbCl5"
ab_step_time.prepare_fastmace_hybrid()
name, atoms = ab_step_time.read_structure(spec, None)
planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
hardware = planner.hardware()
field = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64")
plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0), hardware)
engine = planner.build(plan, field, atoms.copy(), temperature_K=300.0)
for _ in range(3):
    engine.force_field(engine.state.positions)
torch.cuda.synchronize()
print("fp64_emulation", plan.engine.fp64_emulation, flush=True)
torch.cuda.cudart().cudaProfilerStart()
engine.force_field(engine.state.positions)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
