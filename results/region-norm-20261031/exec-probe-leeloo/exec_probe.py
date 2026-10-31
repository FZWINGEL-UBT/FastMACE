"""Fix 2 measurement: what the captured graphs hold outside the caching allocator, per graph key.

Builds the default float32 Simulation (as memory_ledger.py), runs warm-up and steady steps, then drops the graphs one
key at a time (B keys first, then the rest) and measures cudaMemGetInfo minus the allocator's reserved bytes after
each drop. Measurement only.
    python exec_probe.py FM_SOURCE Structure NXxNYxNZ OUTPUT [FIELD=VALUE ...]
"""

import gc
import importlib.util
import json
import sys
from pathlib import Path

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM / "tools" / "bench"))
sys.path.insert(0, str(FM / "tools"))
import fastmace as fm  # noqa: E402  (before torch)
from fastmace.model.compile import configure_allocator  # noqa: E402

# isort: split
import capacity_scan  # noqa: E402

configure_allocator()
import torch  # noqa: E402


def point(label):
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    row = dict(label=label, outside_GiB=(total - free - reserved) / 2**30, reserved_GiB=reserved / 2**30,
               free_GiB=free / 2**30)
    print(f"{label:60s} outside {row['outside_GiB']:.3f} reserved {row['reserved_GiB']:.3f} free {row['free_GiB']:.3f}",
          flush=True)
    return row


structure, repeat_text, output = sys.argv[2], sys.argv[3], Path(sys.argv[4])
rewrites = dict(item.split("=", 1) for item in sys.argv[5:])
name, cell = capacity_scan.base_cell(structure)
atoms = cell.repeat(tuple(int(n) for n in repeat_text.split("x")))
atoms.set_pbc(True)
spec = importlib.util.spec_from_file_location("_ab_frontends", FM / "tools" / "ab_frontends.py")
frontends = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontends)
frontends.thermalize(atoms, 300.0, 1)
from fastmace.config import RewriteConfig  # noqa: E402

rows = [point("start")]
field = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32", rewrites=RewriteConfig(**rewrites))
sim = fm.Simulation(atoms, field, seed=1, integrator=fm.integrators.Langevin(
    timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002))
torch.cuda.synchronize()
rows.append(point("built"))
engine = sim._engine
sim.run(steps=40)
torch.cuda.synchronize()
rows.append(point("40 steps"))
keys = list(engine._graphs)
info = dict(structure=name, atoms=len(atoms), repeat=repeat_text, rewrites=rewrites, gpu=torch.cuda.get_device_name(),
            graphs=[str(k) for k in keys], edges=int(engine.force_field.edge_count),
            padded=sorted(int(e) for e in engine.edge_sizes_used), captures=list(engine.capture_memory),
            invalidations=list(engine.invalidations), recaptures=[str(r) for r in getattr(engine, "recapture_log", [])])
print(json.dumps({k: info[k] for k in ("graphs", "edges", "padded")}), flush=True)
order = sorted(keys, key=lambda k: 0 if isinstance(k, tuple) else 1)
for key in order:
    engine.invalidate_graphs("exec probe", keys=[key])
    gc.collect()
    torch.cuda.synchronize()
    rows.append(point(f"dropped {key}"))
engine.invalidate_graphs("exec probe: all")
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
rows.append(point("all dropped, cache emptied"))
info["points"] = rows
output.mkdir(parents=True, exist_ok=True)
(output / f"exec-{name}-{len(atoms)}.json").write_text(json.dumps(info, indent=1))
