"""Float64 table fit and certificate on MH-1, every pair and interaction, for several fit tolerances (CPU certify)."""
import sys
import time
from pathlib import Path

import fastmace as fm

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402
import torch  # noqa: E402

from fastmace.config import EngineConfig, IntegratorConfig  # noqa: E402
from fastmace.planner import minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402
from fastmace.rewrites import radial  # noqa: E402

ab_step_time.prepare_fastmace_hybrid()
name, atoms = ab_step_time.read_structure(sys.argv[1], None)
planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
hw = planner.hardware()
field = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64")
plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0), hw)
engine = planner.build(plan, field, atoms.copy(), temperature_K=300.0)
model = engine.force_field.model
numbers = sorted(set(atoms.get_atomic_numbers().tolist()))
columns = [model.atomic_numbers.tolist().index(z) for z in numbers]
pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))]
for tol in [float(x) for x in sys.argv[2:]]:
    radial.TOLERANCE_64 = tol
    for index, interaction in enumerate(model.interactions):
        t0 = time.perf_counter()
        chain64 = radial._Chain(model, interaction, columns, torch.float64, torch.device("cpu"))
        tables = radial.fit(chain64, pairs, float(model.r_max), table_dtype=torch.float64)
        table = radial._table_module(tables, columns, torch.device("cuda"))
        net = radial._Chain(model, interaction, columns, torch.float64, torch.device("cuda"))
        cert = radial.certify(table, chain64, net, pairs, float(model.r_max), seed=index)
        print(f"{name} tol {tol:.0e} interaction {index}: pieces {tables['coef'].shape[1]} certificate "
              f"{cert['worst']:.2f} largest {cert['largest_error']:.1e} ({time.perf_counter() - t0:.0f} s)", flush=True)
