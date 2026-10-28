"""ideas D1 on a GPU: does the planner apply ``EngineConfig.fp64_emulation_mantissa_bits=56`` there?

Plans and builds the float64 package for NaCl with ``fp64_emulation_mantissa_bits=56`` and prints the
``fp64_emulation`` decision, the cuBLAS emulation variables in the process afterwards and whether float64 products
run emulated. On full-rate float64 GPUs (A100) the measured rule leaves the emulation off, so the mantissa setting
is never read. Prints one JSON line.
"""
import json
import os
import sys
from pathlib import Path

import fastmace as fm  # before torch: soft file locks

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402
import torch  # noqa: E402

from fastmace.config import EngineConfig, IntegratorConfig  # noqa: E402
from fastmace.model import compile as compiling  # noqa: E402
from fastmace.planner import minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

ab_step_time.prepare_fastmace_hybrid()
name, atoms = ab_step_time.read_structure("NaCl", None)
planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid", fp64_emulation_mantissa_bits=56))
field = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64")
plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0),
                    planner.hardware())
decision = next(d for d in plan.decisions if d[0] == "EngineConfig.fp64_emulation")
engine = planner.build(plan, field, atoms.copy(), temperature_K=300.0)
print(json.dumps(dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                      fp64_emulation=plan.engine.fp64_emulation, decision=list(map(str, decision)),
                      mantissa_bits_requested=plan.engine.fp64_emulation_mantissa_bits,
                      cublas_environment={k: v for k, v in os.environ.items() if k.startswith("CUBLAS")},
                      emulated_products=compiling.fp64_emulation_active(engine.state.positions.device))))
