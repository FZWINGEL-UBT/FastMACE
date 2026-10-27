"""Print the atomic operations of the vendored streaming kernels' CUDA source (MH-1 NaCl problem)."""
import re

import fastmace  # noqa: F401
from openequivariance.core.e3nn_lite import TPProblem

from common import load
from fastmace.kernels.oeq_streaming.runtime import render_kernels

r = load("ops-mh1-r2scan-NaCl.pt")
p = r["problem"]
problem = TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                    [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                     for i in p["instructions"]],
                    irrep_normalization="none", path_normalization="none", shared_weights=False, internal_weights=False)
_, kernel = render_kernels(problem, layout=p["layout"])
src = kernel.jit_kernel
open("streaming_src.cu", "w").write(src)
functions = [(m.start(), m.group(1)) for m in re.finditer(r"__global__[^(]*?(\w+)\s*\(", src)]
for number, line in enumerate(src.splitlines(), 1):
    if "atomic" in line:
        print(number, line.strip()[:150])
print(len(src.splitlines()), "lines;", [name for _, name in functions])
