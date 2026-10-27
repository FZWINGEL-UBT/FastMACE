"""Fit and certify float64 tables on MH-1 (default float64 hybrid plan, tables off), on the CPU and on CUDA."""
import sys
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
print("model dtype", next(model.parameters()).dtype)
columns = sorted({int(c) for c in engine.force_field.data["node_attrs"].argmax(1).tolist()}) if hasattr(
    engine.force_field, "data") and "node_attrs" in getattr(engine.force_field, "data", {}) else list(
    range(len(model.atomic_numbers)))
print("columns", columns[:10], len(columns))
interaction = model.interactions[0]
pairs = [(0, 0)]
cols = columns[:1] if len(columns) > 2 else columns
chain64 = radial._Chain(model, interaction, cols, torch.float64, torch.device("cpu"))
tables = radial.fit(chain64, pairs, float(model.r_max), table_dtype=torch.float64)
print("pieces", tables["coef"].shape, "fit_error", tables["fit_error"])
for device in ("cpu", "cuda"):
    table = radial._table_module(tables, cols, torch.device(device))
    chain_net = radial._Chain(model, interaction, cols, torch.float64, torch.device(device))
    cert = radial.certify(table, chain64, chain_net, pairs, float(model.r_max))
    print(device, "worst", cert["worst"], "largest", cert["largest_error"])
    r = torch.linspace(radial.R_LO_64, float(model.r_max) - 1e-6, 2001, dtype=torch.float64)
    tab, dtab = radial._table_values(table, r.to(device), torch.zeros(r.shape, dtype=torch.int32, device=device))
    truth, dtruth = chain64.with_derivative(r, 0, 0)
    net, dnet = chain_net.with_derivative(r.to(device), 0, 0)
    for label, a in (("table", tab), ("net", net)):
        err = ((a.cpu() - truth).abs() / truth.abs().amax(0)).amax(1)
        print(" ", label, "value err max %.2e at r=%.3f" % (err.max(), r[err.argmax()]))
    for label, a in (("table", dtab), ("net", dnet)):
        err = ((a.cpu() - dtruth).abs() / dtruth.abs().amax(0)).amax(1)
        print(" ", label, "slope err max %.2e at r=%.3f" % (err.max(), r[err.argmax()]))
