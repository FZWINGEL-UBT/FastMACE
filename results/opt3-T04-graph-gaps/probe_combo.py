"""OPT3-T04 probe: NaCl's default plan with Inductor's combo kernels (pointwise only) or without.

    python probe_combo.py OUT.json [--combo]

One process per side (the global Inductor config must be set before the first compile). Prints the
force-graph kernels per replay (profiler, 10 replays), the gaps inside a replay, and the un-profiled step time
over 1000 steps after 300 warm-up steps. Positions after a fixed 50-step run from the same seed are saved,
to compare the two sides.
"""

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import json
import os
import sys
import time

import numpy as np
import torch

combo = "--combo" in sys.argv
if combo:
    import torch._inductor.config as inductor

    inductor.combo_kernels = True
    inductor.combo_kernels_pointwise_only = True

from ase.io import read  # noqa: E402

from fastmace.config import EngineConfig, IntegratorConfig  # noqa: E402
from fastmace.model import loading  # noqa: E402
from fastmace.model.forcefield import ForceField  # noqa: E402
from fastmace.planner import minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

base = read(os.path.join(os.environ["FASTMACE_STRUCTURES"], "Structures/Benchmark/R2SCAN/NaCl.cif")).repeat((3, 3, 3))
base.set_pbc(True)
planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
hardware = planner.hardware()
loaded = loading.load_pristine("mh1-r2scan", checkpoint=os.path.expanduser("~/models/mace-mh-1.model"))
ff = ForceField(loaded, base, precision="float32")
integrator = IntegratorConfig(temperature_K=1000.0, seed=20261021)
plan = planner.plan(ff, base, integrator, Workload(max_temperature_K=1000.0), hardware)
engine = planner.build(plan, ff, base.copy(), temperature_K=1000.0)
for _ in range(50):
    engine.step()
engine.flush()
positions = engine.state.positions.detach().cpu().numpy().copy()
for _ in range(250):
    engine.step()
engine.flush()
torch.cuda.synchronize()
started = time.perf_counter()
for _ in range(1000):
    engine.step()
engine.flush()
torch.cuda.synchronize()
step_ms = (time.perf_counter() - started)
keys = [key for key in engine._graphs if isinstance(key, tuple) and key[0] == "B"]
graph = engine._graphs[keys[-1]]
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
kernels = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                 key=lambda e: e.time_range.start)
names = [e.name for e in kernels]
per = len(kernels) // 10
gaps = 0.0
for r in range(10):
    chunk = kernels[r * per:(r + 1) * per]
    gaps += sum(max(0.0, b.time_range.start - a.time_range.end) for a, b in zip(chunk, chunk[1:]))
out = {"combo": combo, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
       "fastmace_commit": open(os.path.join(os.environ["PYTHONPATH"], ".source_commit")).read().strip(),
       "step_ms": step_ms, "kernels_per_replay": per, "gap_us_per_replay": gaps / 10,
       "combo_kernels": sum("combo" in n for n in names[:per]), "b_keys": [str(k) for k in keys]}
np.save(sys.argv[1].replace(".json", "-positions.npy"), positions)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out))
