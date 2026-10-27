"""Compile-only check: render the float64 streaming source for a recorded MH-1 problem, transform, compile (sm_120)."""
import sys

import numpy as np
from openequivariance.core.e3nn_lite import TPProblem

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ddconv import dd_forward_source  # noqa: E402
from fastmace.kernels.oeq_streaming.runtime import compile_cubin, render_kernels  # noqa: E402

record = torch.load(sys.argv[1])["records"][0]
p = record["problem"]
problem = TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                    [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                     for i in p["instructions"]],
                    irrep_normalization="none", path_normalization="none", shared_weights=False,
                    internal_weights=False)
_, kernel = render_kernels(problem, np.float64, 128, p["layout"])
source = kernel.jit_kernel
open("native_f64_source.cu", "w").write(source)
for convert in ("cvt", "bits"):
    dd = dd_forward_source(source, convert)
    open(f"dd_forward_{convert}.cu", "w").write(dd)
    cubin = compile_cubin(dd, (12, 0), ("--fmad=false",))
    open(f"dd_forward_{convert}.cubin", "wb").write(cubin)
    print(convert, "compiled", len(cubin), "bytes")
cubin = compile_cubin(source, (12, 0))
open("native_f64.cubin", "wb").write(cubin)
print("native compiled", len(cubin), "bytes")
