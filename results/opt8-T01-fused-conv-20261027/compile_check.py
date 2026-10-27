"""CPU-only: generate the fused kernels for the MH-1 problem and compile them with NVRTC for sm_120 (no GPU use)."""
import sys

sys.path.insert(0, "/mnt/d/Margraf/fm-opt8/tools/bench")
import fastmace  # noqa: F401
import numpy as np
import torch
from openequivariance.core.e3nn_lite import TPProblem

import opt8_fused_conv as fc
import opt8_fused_kernels as fk
from fastmace.kernels.oeq_streaming.runtime import compile_cubin, render_kernels

p = torch.load("/home/fzwin/tmp/opt7/ops-mh1-r2scan-NaCl.pt")["records"][0]["problem"]
problem = TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                    [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                     for i in p["instructions"]],
                    irrep_normalization="none", path_normalization="none", shared_weights=False, internal_weights=False)
schedule, kernel = render_kernels(problem, np.float32, 128, p["layout"])
for path in schedule.paths:
    print(path.input_start, path.input_irrep_dim, path.edge_start, path.edge_irrep_dim, path.weight_start,
          [(o.accumulator.name, len(o.terms)) for o in path.outputs])
print("terms", sum(len(o.terms) for pa in schedule.paths for o in pa.outputs))


class FakeModule:
    def __init__(self, source, names, extra_options=()):
        cubin = compile_cubin(source, (12, 0), list(extra_options))
        print("cubin", len(cubin))

    def set_shared(self, *a):
        pass


cfg = dict(kv.split("=") for kv in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if kv)
f = fk.FusedConv(schedule, "ir_mul", torch.zeros(1280, 65), cfg, module_cls=FakeModule, sms=36)
open("/home/fzwin/tmp/opt8/fused_src.cu", "w").write(f.src)
print(f.describe())
