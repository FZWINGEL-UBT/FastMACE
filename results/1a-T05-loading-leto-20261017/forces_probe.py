"""Forces of MH-1 (head matpes_r2scan) on rattled NaCl 2x2x2 from FastMACE's loader or the baseline's.

    python forces_probe.py fastmace|baseline cueq|hybrid OUT.json [BASELINE_ROOT]

fastmace: ``fastmace.model.loading.load`` (FASTMACE_MODELS gives the checkpoint;
the hybrid prepares the OEq runtime in-process). baseline: the research
engine's ``common.load_calculator`` path as ``workflow.load_md_calculator`` drives
it, from an export of tag research-engine-baseline-20261017 (the hybrid must
run under that export's oeq_no_nvcc.sh). cueq runs float64 with the fused
convolution (main.py's --cueq-conv-fusion default), hybrid float32.
"""

import json
import os
import platform
import sys
import time

route, backend, out = sys.argv[1:4]
dtype = "float64" if backend == "cueq" else "float32"
start = time.perf_counter()
if route == "fastmace":
    import fastmace  # noqa: F401  (before torch)
    from fastmace.model import loading
    loaded = loading.load("mh1-r2scan", backend=backend, dtype=dtype, device="cuda:0")
    calc = loaded.calculator
    extra = {"sha256": loaded.sha256, "trust": loaded.trust, "licence": loaded.licence,
             "cueq_conv_fusion": loaded.cueq_conv_fusion, "fused_cueq": loaded.fused_cueq,
             "patches": [r.name for r in loaded.patches]}
else:
    root = sys.argv[4]
    os.environ.update(MACE_BENCH_DEVICE="cuda:0", MACE_BENCH_DTYPE=dtype, MACE_BENCH_CUEQ="1",
                      MACE_BENCH_CHECKPOINT=os.path.join(os.environ["FASTMACE_MODELS"], "mace-mh-1.model"))
    sys.path[:0] = [os.path.join(root, "Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts"),
                    os.path.join(root, "Experiments/Langevin_Thermostat")]
    from langevin import upstream_mace  # noqa: E402  (imports langevin before torch: soft locks)
    import common  # noqa: E402
    upstream_mace.use_cueq_conv_fusion(True if backend == "cueq" else None)
    calc = common.load_calculator("mh1-r2scan", dtype=dtype, enable_cueq=True, enable_oeq=backend == "hybrid")
    upstream_mace.require_fused_cueq(calc.models[0])
    extra = {}
load_s = time.perf_counter() - start

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase.build import bulk  # noqa: E402

atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
atoms.rattle(0.05, seed=17)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
record = {"route": route, "backend": backend, "dtype": dtype, "host": platform.node(),
          "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": sys.executable,
          "load_s": load_s, "energy_eV": float(energy), "forces_eV_A": np.asarray(forces).tolist(),
          "conv_tp": sorted({type(i.conv_tp).__name__ for i in calc.models[0].interactions}), **extra}
with open(out, "w") as handle:
    json.dump(record, handle, indent=1)
print(json.dumps({k: v for k, v in record.items() if k != "forces_eV_A"}))
